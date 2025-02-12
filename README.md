# AgentDsl

# VERY EXPERIMENTAL do not use anywhere!
### API will change, every day, maybe even two times a day. 

### Backend API 

- Replicate
- Gemini
- OpenAi

need keys for each API, add to .env

```bash
export REPLICATE_API_KEY=xxxx
export GEMINI_API_KEY=xxxx
export OPENAI_API_KEY=xxxx
```

### Example DSL

```elixir 
defmodule MyAgent do
  use Agent.DSL
  alias Agent.API.{Replicate, Gemini, OpenAi}

  agent do
    initial_state(:ask_intro)
    initial_data(%{name: "Tao"})

    state :ask_intro do
      prompt("""
      Ask user what job he want to create, using this information: <%= data["name"] %>}
      """)

      backend(API.Replicate)

      on_enter do
        ask(%{}, prompt)
      end

      on_event :cast, {:user_input, user_input} do
        run_task(extract_title(user_input))
        run_task(extract_tags(user_input))

        classify check_if_user_write_all(user_input) do
          ":repeat" -> {:keep_state, data}
          ":stop" -> {:stop, data}
          _ -> ask(%{}, prompt)
        end
      end

      handle :info, {:task, :extract_tags, extracted} do
        {:next_state, Map.put(data, :extracted, extracted)}
      end
    end

    classification :check_if_user_write_all, user_input do
      backend(API.Gemini)

      prompt("""
      Classify this input: <%= user_input %>

      answer if user want to repeat -> :repeat
      answer if user want to go next step -> :next
      answer if user want to stop -> :stop
      """)
    end

    task :extract_title, user_input do
      prompt("""
      Extract title from this information: <%= user_input %>
      """)

      backend(API.Gemini)
    end

    task :extract_tags, user_input do
      prompt("""
      Extract tags from this information: <%= user_input %>
      """)

      backend(API.OpenAi)
    end
  end
end
```
