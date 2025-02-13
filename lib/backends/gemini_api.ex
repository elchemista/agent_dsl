defmodule Consulente.Agents.API.Gemini do
  @moduledoc """
  Provides a similar interface as `Consulente.Agents.ReplicateAPI` but for Gemini's (Google) Chat Completions API,
  using HTTPoison for HTTP requests and streaming.
  """

  @gemini_url "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
  @default_model "gemini-1.5-flash-8b-latest"
  @model_flash "gemini-2.0-flash-lite-preview"

  require Logger

  @doc """
  Synchronous call to the Gemini Chat Completions API (no streaming).

  - Takes a `prompt` and optionally a `model` name.
  - Returns `{:ok, decoded_json}` on success, or an `{:error, reason}` on failure.
  """
  def call(prompt, model \\ @default_model) do
    token = System.get_env("GEMINI_API_KEY")

    body =
      Jason.encode!(%{
        model: model,
        messages: [%{role: "user", content: prompt}],
        stream: false
      })

    headers = [
      {"Content-Type", "application/json"},
      {"Authorization", "Bearer #{token}"}
    ]

    with {:ok, resp} <- HTTPoison.post(@gemini_url, body, headers, timeout: 120_000),
         {:ok, decoded} <- Jason.decode(resp.body) do
      {:ok, decoded}
    else
      error -> error
    end
  end

  @doc """
  Stub function to mirror `get_result/1` from ReplicateAPI.
  Gemini returns the full response on `call/2`, so there's no separate resource to retrieve.

  Example usage after you do:

      {:ok, response} = call("Hello world")
      get_result(response)
  """
  def get_result(%{"choices" => choices}) when is_list(choices) do
    # Extract the assistant's message content from each choice
    Enum.map(choices, fn
      %{"message" => %{"content" => content}} -> content
      _ -> ""
    end)
  end

  def get_result(_other), do: ""

  @doc """
  Asynchronous, streaming call.

  Returns a tuple: `{"started", request_id, pid}`
  where `pid` is the Task handling the stream.

  The parent process will receive messages of the form:
      {:stream_token, request_id, partial_content}
  for each partial chunk, and:
      {:stream_done, request_id, leftover_or_done_message}
  upon completion.

  In case of errors, you may receive:
      {:stream_error, request_id, reason}
  """
  def stream(request_id, prompt, model \\ @default_model) do
    parent = self()
    token = System.get_env("GEMINI_API_KEY")

    body =
      Jason.encode!(%{
        model: model,
        messages: [%{role: "user", content: prompt}],
        stream: true
      })

    headers = [
      {"Content-Type", "application/json"},
      {"Authorization", "Bearer #{token}"}
    ]

    # Spawn a Task to handle the streaming POST
    with {:ok, pid} <-
           Task.start_link(fn ->
             case HTTPoison.post(@gemini_url, body, headers,
                    stream_to: self(),
                    recv_timeout: :infinity
                  ) do
               {:ok, _response} ->
                 # We begin receiving and processing chunks in a loop
                 loop_stream(parent, request_id, "")

               {:error, reason} ->
                 send(parent, {:stream_error, request_id, reason})
             end
           end) do
      {"started", request_id, pid}
    else
      error -> error
    end
  end

  # ---------------------------------------------------------------------------
  # Private: receives streaming messages and splits them into SSE-like events.
  # ---------------------------------------------------------------------------
  defp loop_stream(parent, request_id, buffer) do
    receive do
      # The server has sent us a status code (e.g. 200)
      %HTTPoison.AsyncStatus{code: _status_code} ->
        loop_stream(parent, request_id, buffer)

      # The server has sent us headers
      %HTTPoison.AsyncHeaders{headers: _headers} ->
        loop_stream(parent, request_id, buffer)

      # We got a chunk of the SSE data
      %HTTPoison.AsyncChunk{chunk: chunk} ->
        new_buffer = buffer <> chunk
        {events, leftover} = split_events(new_buffer)

        Enum.each(events, &handle_event(&1, parent, request_id))

        loop_stream(parent, request_id, leftover)

      # The request/response has ended
      %HTTPoison.AsyncEnd{} ->
        send(parent, {:stream_done, request_id, buffer})

      # Any other message
      _other ->
        loop_stream(parent, request_id, buffer)
    end
  end

  # Split on double newlines (SSE blocks).
  defp split_events(buffer) do
    parts = Regex.split(~r/\r?\n\r?\n/, buffer, trim: false)

    case parts do
      [single] ->
        # Not enough for a complete event yet
        {[], single}

      many ->
        # All but the last are complete events
        {Enum.slice(many, 0..-2//1), List.last(many)}
    end
  end

  # Handles a single SSE-like event block. Look for lines beginning with `data: `.
  defp handle_event(block, parent, request_id) do
    lines = String.split(block, "\n", trim: true)
    data_lines = Enum.filter(lines, &String.starts_with?(&1, "data:"))

    for line <- data_lines do
      raw = line |> String.replace_prefix("data:", "") |> String.trim()

      case raw do
        "[DONE]" ->
          # End of stream signal
          send(parent, {:stream_done, request_id, "[DONE]"})
          Process.exit(self(), :normal)

        _ ->
          # Otherwise decode JSON and forward
          case Jason.decode(raw) do
            {:ok, %{"choices" => choices}} ->
              # Typically:
              #   "choices": [
              #     {"delta": {"content": "partial text"}}
              #   ]
              Enum.each(choices, fn
                %{"delta" => %{"content" => content}} ->
                  send(parent, {:stream_token, request_id, content})

                other ->
                  # If there's no "content", forward the raw chunk or handle differently
                  send(parent, {:stream_token, request_id, other})
              end)

            {:error, reason} ->
              send(parent, {:stream_error, request_id, reason})
          end
      end
    end
  end
end
