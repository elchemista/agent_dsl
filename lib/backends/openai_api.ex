defmodule Consulente.Agents.API.OpenAi do
  @moduledoc """
  Provides a similar interface as `Consulente.Agents.ReplicateAPI` but for OpenAI's Chat Completions API,
  using HTTPoison for HTTP requests and streaming.
  """

  @openai_url "https://api.openai.com/v1/chat/completions"
  @default_model "gpt-4o-mini"

  require Logger

  @doc """
  Synchronous call to OpenAI Chat Completions API (no streaming).
  Takes a prompt and optionally a model name.
  Returns `{:ok, decoded_response}` on success or an error tuple on failure.
  """
  def call(prompt, model \\ @default_model) do
    token = System.get_env("OPENAI_API_KEY")

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

    with {:ok, resp} <- HTTPoison.post(@openai_url, body, headers, timeout: 120_000),
         {:ok, decoded} <- Jason.decode(resp.body) do
      {:ok, decoded}
    else
      error -> error
    end
  end

  @doc """
  Extracts the assistant's message content from an OpenAI response.

  Usage:
      {:ok, response} = call("Hello world")
      get_result(response)

  or
      get_result(%{"choices" => [...]})
  """
  def get_result(%{"choices" => choices}) do
    choices
    |> Enum.map_join(fn
      %{"message" => %{"content" => content}} -> content
      _ -> ""
    end)
  end

  @doc """
  Asynchronous, streaming call. Returns a tuple:
      {"started", request_id, pid}
  where `pid` is the Task handling the stream. The parent process
  will receive messages of the form:
      {:stream_token, request_id, token_content}
  for each partial chunk, and
      {:stream_done, request_id, leftover_buffer_or_done}
  upon completion.
  In case of errors, you may receive:
      {:stream_error, request_id, reason}
  """
  def stream(request_id, prompt, model \\ @default_model) do
    parent = self()
    token = System.get_env("OPENAI_API_KEY")

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

    # Spawn a Task that will do the streaming POST and handle the SSE-like chunks
    with {:ok, pid} <-
           Task.start_link(fn ->
             case HTTPoison.post(@openai_url, body, headers,
                    stream_to: self(),
                    recv_timeout: :infinity
                  ) do
               {:ok, _async_response} ->
                 # Start receiving and processing chunks
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

  # --------------------------------------------------------------------------
  # Internal: receives streaming messages until HTTPoison.AsyncEnd
  # --------------------------------------------------------------------------
  defp loop_stream(parent, request_id, buffer) do
    receive do
      # The server has sent us a status code (e.g. 200)
      %HTTPoison.AsyncStatus{code: _status_code} ->
        loop_stream(parent, request_id, buffer)

      # The server has sent us headers
      %HTTPoison.AsyncHeaders{headers: _headers} ->
        loop_stream(parent, request_id, buffer)

      # A chunk of data came in
      %HTTPoison.AsyncChunk{chunk: chunk} ->
        new_buffer = buffer <> chunk
        {events, leftover} = split_events(new_buffer)
        Enum.each(events, &handle_event(&1, parent, request_id))
        loop_stream(parent, request_id, leftover)

      # The request/response has ended
      %HTTPoison.AsyncEnd{} ->
        send(parent, {:stream_done, request_id, buffer})

      # Any other message we might get
      other ->
        # You can pattern match or log if needed
        loop_stream(parent, request_id, buffer)
    end
  end

  # --------------------------------------------------------------------------
  # Splits the buffer into SSE-like events
  # (chunks separated by a double newline "\n\n" or "\r\n\r\n").
  # Returns {complete_events, leftover_buffer}.
  # --------------------------------------------------------------------------
  defp split_events(buffer) do
    parts = Regex.split(~r/\r?\n\r?\n/, buffer, trim: false)

    case parts do
      [single] ->
        # No full event in 'single' yet
        {[], single}

      many ->
        # All but the last piece are complete events
        {Enum.slice(many, 0..-2//1), List.last(many)}
    end
  end

  # --------------------------------------------------------------------------
  # Handles an individual SSE-like block of data from OpenAI.
  # Each block can contain one or more lines that look like "data: {...}"
  # or "data: [DONE]".
  # --------------------------------------------------------------------------
  defp handle_event(block, parent, request_id) do
    lines = String.split(block, "\n", trim: true)
    data_lines = Enum.filter(lines, &String.starts_with?(&1, "data:"))

    for dline <- data_lines do
      raw = dline |> String.replace_prefix("data:", "") |> String.trim()

      case raw do
        "[DONE]" ->
          # The server signals it is done
          send(parent, {:stream_done, request_id, "[DONE]"})
          Process.exit(self(), :normal)

        _ ->
          case Jason.decode(raw) do
            {:ok, %{"choices" => choices}} ->
              Enum.each(choices, fn choice ->
                case choice["delta"] do
                  %{"content" => content} ->
                    send(parent, {:stream_token, request_id, content})

                  # Sometimes the "delta" might have "role": "assistant", or be empty.
                  # You can handle that case if you need it.
                  _ ->
                    :ok
                end
              end)

            {:error, reason} ->
              send(parent, {:stream_error, request_id, reason})
          end
      end
    end
  end
end
