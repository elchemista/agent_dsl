defmodule Consulente.Agents.API.Replicate do
  @replicate_url "https://api.replicate.com/v1/models"
  @default_model "meta/meta-llama-3-8b-instruct"

  require Logger

  # Synchronous call: performs a normal POST (stream: false) and then uses the "get" URL.
  def call(prompt, model \\ @default_model) do
    token = System.get_env("REPLICATE_API_KEY")
    body = Jason.encode!(%{input: %{prompt: prompt}, stream: false})

    headers = [
      {"Content-Type", "application/json"},
      {"Authorization", "Bearer #{token}"}
    ]

    with {:ok, resp} <-
           HTTPoison.post("#{@replicate_url}/#{model}/predictions", body, headers,
             timeout: 120_000
           ),
         {:ok, decoded} <- Jason.decode(resp.body),
         %{"urls" => urls} <- decoded,
         get_url when is_binary(get_url) <- Map.get(urls, "get"),
         {:ok, final_resp} <- HTTPoison.get(get_url, headers),
         {:ok, final_decoded} <- Jason.decode(final_resp.body) do
      {:ok, final_decoded}
    else
      error -> error
    end
  end

  def get_result(list) when is_list(list), do: list

  def get_result(%{"urls" => %{"get" => url}}) do
    get_result(url)
  end

  def get_result(url) when is_bitstring(url) do
    token = System.get_env("REPLICATE_API_KEY")

    headers = [
      {"Content-Type", "application/json"},
      {"Authorization", "Bearer #{token}"}
    ]

    with {:ok, resp} <- HTTPoison.get(url, headers, timeout: 120_000),
         {:ok, decoded} <- Jason.decode(resp.body) do
      Map.get(decoded, "output")
      |> case do
        nil -> decoded
        list when is_list(list) -> Enum.join(list)
      end
    end
  end

  # Asynchronous call: performs a POST with stream: true, then spawns a Task to stream SSE tokens.
  def stream(id, prompt, model \\ @default_model) do
    parent = self()
    token = System.get_env("REPLICATE_API_TOKEN")
    body = Jason.encode!(%{input: %{prompt: prompt}, stream: true})

    headers = [
      {"Content-Type", "application/json"},
      {"Authorization", "Bearer #{token}"}
    ]

    with {:ok, resp} <- HTTPoison.post("#{@replicate_url}/#{model}/predictions", body, headers),
         {:ok, decoded} <- Jason.decode(resp.body),
         %{"urls" => urls} <- decoded,
         stream_url when is_binary(stream_url) <- Map.get(urls, "stream"),
         {:ok, pid} <-
           Task.start_link(fn -> stream_loop({stream_url, token}, {parent, id}, "") end) do
      {decoded["status"], id, pid}
    else
      error -> error
    end
  end

  # Stream loop: connects to the SSE endpoint and processes chunks.
  defp stream_loop({url, token}, {parent, id}, buffer) do
    headers = [
      {"Accept", "text/event-stream"},
      {"Authorization", "Bearer #{token}"}
    ]

    case HTTPoison.get(url, headers, stream_to: self(), recv_timeout: :infinity) do
      {:ok, _} -> loop({parent, id}, buffer)
      {:error, reason} -> send(parent, {:stream_error, id, reason})
    end
  end

  defp loop(task = {parent, id}, buffer) do
    receive do
      %HTTPoison.AsyncChunk{chunk: chunk} ->
        new_buffer = buffer <> chunk
        {events, leftover} = split_events(new_buffer)
        Enum.each(events, &handle_event(&1, task))
        loop(task, leftover)

      %HTTPoison.AsyncEnd{} ->
        send(parent, {:stream_done, id, buffer})

      _other ->
        loop(task, buffer)
    after
      1000 ->
        loop(task, buffer)
    end
  end

  # Splits the buffer on "\n\n"; returns {complete_events, leftover}
  defp split_events(buffer) do
    # Split the buffer on double newlines (handling LF and CRLF)
    parts = Regex.split(~r/\r?\n\r?\n/, buffer, trim: false)

    case parts do
      [single] -> {[], single}
      many -> {Enum.slice(many, 0..-2//1), List.last(many)}
    end
  end

  # Parses an SSE event block and sends tokens.
  defp handle_event(block, {parent, id}) do
    lines = String.split(block, "\n", trim: true)

    event =
      case Enum.find(lines, &String.starts_with?(&1, "event:")) do
        nil ->
          "data"

        line ->
          String.replace_prefix(line, "event:", "") |> remove_first_space()
      end

    data_lines = Enum.filter(lines, &String.starts_with?(&1, "data:"))

    Enum.each(data_lines, fn dline ->
      token_data = String.replace_prefix(dline, "data:", "") |> remove_first_space()

      if String.downcase(event) == "DONE" do
        send(parent, {:stream_done, id, token_data})
        Process.exit(self(), :normal)
      else
        send(parent, {:stream_token, id, token_data})
      end
    end)
  end

  defp remove_first_space(" " <> rest), do: rest
  defp remove_first_space(rest), do: rest
end
