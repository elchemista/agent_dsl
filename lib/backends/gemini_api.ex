defmodule Agent.API.Gemini do
  @moduledoc """
  Provides a similar interface as `Consulente.Agents.ReplicateAPI` but for OpenAI's Chat Completions API,
  using HTTPoison for HTTP requests and streaming.
  """

  @openai_url "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
  @default_model "gemini-1.5-flash-8b-latest"
  @model_flash "gemini-2.0-flash-lite-preview"

  require Logger

  @doc """
  Synchronous call to OpenAI Chat Completions API (no streaming).
  Takes a prompt and optionally a model name.
  Returns `{:ok, content}` on success or an error tuple on failure.
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

    with {:ok, resp} <- HTTPoison.post(@openai_url, body, headers, timeout: 120_000),
         {:ok, decoded} <- Jason.decode(resp.body) do
      # Extract the assistant's message content from the first choice
      {:ok, decoded}
    else
      error -> error
    end
  end

  @doc """
  Stub function to mirror the `get_result/1` from ReplicateAPI.
  OpenAI returns the full response immediately on `call/2`,
  so there's no separate resource to retrieve.
  """
  def get_result(%{"choices" => choices}) do
    # Extract the assistant's message content from the first choice
    choices
    |> Enum.map(fn %{"message" => %{"content" => content}} -> content end)
  end

  @doc """
  Asynchronous, streaming call. Returns a tuple:

      {"started", "no-id", pid}

  where `pid` is the Task handling the stream. The parent process
  will receive messages of the form:

      {:stream_token, <json-decoded-chunk>}

  for each partial chunk, and

      {:stream_done, leftover_or_done_message}

  upon completion.

  In case of errors, you may receive:

      {:stream_error, reason}
  """
  def stream(id, prompt, model \\ @default_model) do
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
           Task.start_link(fn -> stream_loop({@openai_url, body, headers}, {parent, id}, "") end) do
      # We mirror the return structure: {status, id, pid}
      {"started", body["model"] <> "-" <> body["created"], pid}
    else
      error -> error
    end
  end

  # ----------------------------------------------------------------------------
  # Private functions handling streaming logic (similar to ReplicateAPI)
  # ----------------------------------------------------------------------------

  # Makes the POST call in streaming mode; if successful, loops to process chunks.
  defp stream_loop({url, body, headers}, task = {parent, id}, buffer) do
    opts = [
      stream_to: self(),
      async: :once,
      recv_timeout: :infinity
    ]

    case HTTPoison.post(url, body, headers, opts) do
      {:ok, %HTTPoison.AsyncResponse{id: ref}} ->
        # Start processing events
        loop(task, buffer, ref)

      {:error, reason} ->
        send(parent, {:stream_error, reason})
    end
  end

  defp loop(task = {parent, id}, buffer, ref) do
    receive do
      %HTTPoison.AsyncStatus{id: ^ref, code: _status_code} ->
        # Request next chunk
        HTTPoison.stream_next(ref)
        loop(task, buffer, ref)

      %HTTPoison.AsyncHeaders{id: ^ref, headers: _headers} ->
        # Request next chunk
        HTTPoison.stream_next(ref)
        loop(task, buffer, ref)

      %HTTPoison.AsyncChunk{id: ^ref, chunk: chunk} ->
        new_buffer = buffer <> chunk
        {events, leftover} = split_events(new_buffer)

        Enum.each(events, &handle_event(&1, task))

        # Request next chunk
        HTTPoison.stream_next(ref)
        loop(task, leftover, ref)

      %HTTPoison.AsyncEnd{id: ^ref} ->
        # No more data
        send(parent, {:stream_done, id, buffer})

      _other ->
        loop(task, buffer, ref)
    end
  end

  # Split on double newlines (as if SSE); returns {complete_events, leftover}.
  # OpenAI chunked data often looks like:
  #
  # data: {...}\n\n
  # data: {...}\n\n
  # data: [DONE]
  defp split_events(buffer) do
    parts = Regex.split(~r/\r?\n\r?\n/, buffer, trim: false)

    case parts do
      [single] ->
        # No full event found yet
        {[], single}

      many ->
        # All but the last piece are complete
        {Enum.slice(many, 0..-2), List.last(many)}
    end
  end

  # Handle an individual SSE-like block; look for lines beginning with `data: `
  defp handle_event(block, {parent, id}) do
    lines = String.split(block, "\n", trim: true)
    data_lines = Enum.filter(lines, &String.starts_with?(&1, "data:"))

    for dline <- data_lines do
      raw = dline |> String.replace_prefix("data:", "") |> String.trim()

      case raw do
        "[DONE]" ->
          # End of stream
          # keep same as replicate api
          send(parent, {:stream_done, id, "{}"})
          Process.exit(self(), :normal)

        # Otherwise decode JSON and forward
        _ ->
          case Jason.decode(raw) do
            {:ok, %{"choices" => choices}} ->
              # Each chunk typically has:
              # {
              #   "id": ...,
              #   "object": "chat.completion.chunk",
              #   "choices": [
              #     {
              #       "delta": { "content": "partial text" },
              #       "index": 0,
              #       "finish_reason": nil
              #     }
              #   ]
              # }
              Enum.each(
                choices,
                fn
                  %{"delta" => %{"content" => content}} ->
                    send(parent, {:stream_token, id, content})

                  choice ->
                    send(parent, {:stream_token, id, choice})
                end
              )

            {:error, reason} ->
              send(parent, {:stream_error, id, reason})
          end
      end
    end
  end
end
