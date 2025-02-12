defmodule Agent.DSL do
  @moduledoc """
  A DSL for building AI-based gen_statem agents with prompts, tasks, and states.
  """

  defmacro __using__(_opts) do
    quote do
      # We accumulate data in these attributes
      Module.register_attribute(__MODULE__, :agent_initial_state, persist: false)
      Module.register_attribute(__MODULE__, :agent_initial_data, persist: false)
      Module.register_attribute(__MODULE__, :agent_states, accumulate: true)
      Module.register_attribute(__MODULE__, :agent_tasks, accumulate: true)
      Module.register_attribute(__MODULE__, :agent_classifications, accumulate: true)

      @before_compile AgentDSL

      alias Agent.API.{Replicate, Gemini, OpenAi}

      import AgentDSL,
        only: [
          agent: 1,
          initial_state: 1,
          initial_data: 1,
          state: 2,
          classification: 3,
          task: 3,
          prompt: 1,
          backend: 1,
          on_enter: 1,
          handle: 3,
          classify: 2,
          run_task: 1,
          start_sse: 2
        ]
    end
  end

  # -------------------------
  # The top-level agent do ... end
  # -------------------------
  defmacro agent(do: block) do
    quote do
      unquote(block)
    end
  end

  # -------------------------
  # initial_state and initial_data macros
  # -------------------------
  defmacro initial_state(state_name) do
    quote do
      @agent_initial_state unquote(state_name)
    end
  end

  defmacro initial_data(data) do
    quote do
      @agent_initial_data unquote(data)
    end
  end

  # -------------------------
  # state :name do ... end
  # Inside we might define prompt, backend, on_enter, handle ...
  # We'll parse them in __before_compile__
  # -------------------------
  defmacro state(state_name, do: block_ast) do
    quote do
      @agent_states {unquote(state_name), unquote(block_ast)}
    end
  end

  # A macro for "prompt """...""""
  defmacro prompt(str) do
    quote do
      # We'll store this prompt in the current state block.
      # We'll parse it later in __before_compile__.
      {:prompt, unquote(str)}
    end
  end

  # A macro for backend SomeModule
  defmacro backend(mod_ast) do
    quote do
      {:backend, unquote(mod_ast)}
    end
  end

  # on_enter do ... end
  defmacro on_enter(do: block) do
    quote do
      {:on_enter, unquote(block)}
    end
  end

  # handle event_type, pattern do ... end
  defmacro on_event(event_type, pattern_ast, do: block_ast) do
    quote do
      {:handle, unquote(event_type), unquote(pattern_ast), unquote(block_ast)}
    end
  end

  # The "classify check_if_user_write_all(...) do ... end" macro
  # We'll store it separately from states so we can reference it from a handle block
  defmacro classify(classifier_name, do: block_ast) do
    quote do
      {:classify_invocation, unquote(classifier_name), unquote(Macro.escape(block_ast))}
    end
  end

  # If we just want "classify check_if_user_write_all(user_input) do ... end"
  # we can parse it as "check_if_user_write_all(user_input)" so we might
  # want to store the arguments.
  # But for now, let's keep it simple with the above approach.

  # We'll define run_task ...
  defmacro run_task(task_call_ast) do
    quote do
      {:run_task, unquote(Macro.escape(task_call_ast))}
    end
  end

  # If we want start_sse DSL
  defmacro ask(data, prompt_expr) do
    quote do
      {:start_sse, unquote(data), unquote(prompt_expr)}
    end
  end

  # -------------
  # classification and task macros
  # -------------
  defmacro classification(name, arg_ast, do: block_ast) do
    # We'll store that in :agent_classifications
    quote do
      @agent_classifications {unquote(name), unquote(arg_ast), unquote(block_ast)}
    end
  end

  defmacro task(name, arg_ast, do: block_ast) do
    quote do
      @agent_tasks {unquote(name), unquote(arg_ast), unquote(block_ast)}
    end
  end

  # -------------------------
  # BEFORE COMPILE
  # This is where we transform all the stored data into a gen_statem
  # -------------------------
  defmacro __before_compile__(env) do
    initial_state = Module.get_attribute(env.module, :agent_initial_state) || :undefined
    initial_data = Module.get_attribute(env.module, :agent_initial_data) || %{}
    states = Module.get_attribute(env.module, :agent_states) || []
    tasks = Module.get_attribute(env.module, :agent_tasks) || []
    classifications = Module.get_attribute(env.module, :agent_classifications) || []

    # We'll generate a handle_event/4 that pattern matches on state_name, event_type, event_content
    handle_event_ast = build_handle_event(states)

    # tasks and classifications can be turned into helper functions, e.g. defp do_task(name, args)
    tasks_ast = build_tasks_ast(tasks)
    classifications_ast = build_classifications_ast(classifications)

    final_module_ast =
      quote do
        @behaviour :gen_statem

        # The init callback
        @impl true
        def init(_args) do
          data = unquote(Macro.escape(initial_data))
          {:ok, unquote(initial_state), data}
        end

        @impl true
        def callback_mode do
          [:handle_event_function, :state_enter]
        end

        unquote(handle_event_ast)
        unquote(tasks_ast)
        unquote(classifications_ast)

        @impl true
        def terminate(_reason, state, data) do
          IO.puts("Terminating in #{inspect(state)} with data: #{inspect(data)}")
          :ok
        end
      end

    IO.puts("\n=== DEBUG: FINAL EXPANDED CODE for #{inspect(env.module)} ===")
    IO.puts(Macro.to_string(final_module_ast))
    IO.puts("=== END DEBUG ===\n")
  end

  # -------------------------
  # Build handle_event/4 AST
  # -------------------------
  defp build_handle_event(states) do
    # states is a list of {state_name, block_ast}
    # Inside block_ast we have a __block__ of expressions:
    #   {:prompt, "..."}
    #   {:backend, SomeModule}
    #   {:on_enter, ...}
    #   {:handle, event_type, pattern, block_ast} ...
    # We'll parse them out and build up match clauses.

    clauses =
      for {st_name, block_ast} <- states do
        # parse out each item: prompt, backend, on_enter, handle
        {prompt_ast, backend_ast, on_enter_ast, handle_list} = parse_state_block(block_ast)

        # Build the :enter clause
        enter_clause =
          quote do
            {unquote(st_name), :enter, _content} ->
              # We'll inject user on_enter code
              unquote(on_enter_ast)
          end

        # For each handle item, we generate a match clause
        handle_clauses =
          Enum.map(handle_list, fn {:handle, ev_type, pattern, body_ast} ->
            quote do
              {unquote(st_name), unquote(ev_type), unquote(pattern)} ->
                unquote(generate_handle_body(body_ast, prompt_ast, backend_ast))
            end
          end)

        [enter_clause | handle_clauses]
      end

    quote do
      @impl true
      def handle_event(event_type, event_content, state_name, data) do
        case {state_name, event_type, event_content} do
          (unquote_splicing(clauses))
          # _ -> {:keep_state, data}
        end
      end
    end
  end

  # parse_state_block: walk the AST to find :prompt, :backend, :on_enter, and :handle
  defp parse_state_block({:__block__, _, lines}) do
    prompt_ast = nil
    backend_ast = nil

    on_enter_ast =
      quote do
        {:keep_state, data}
      end

    handle_list = []

    {prompt_ast, backend_ast, on_enter_ast, handle_list} =
      Enum.reduce(lines, {prompt_ast, backend_ast, on_enter_ast, handle_list}, fn
        {:prompt, p}, {pa, ba, oe, hl} ->
          {p, ba, oe, hl}

        {:backend, b}, {pa, ba, oe, hl} ->
          {pa, b, oe, hl}

        {:on_enter, block}, {pa, ba, _oe, hl} ->
          {pa, ba, expand_on_enter(block), hl}

        {:handle, event_type, pattern, body}, {pa, ba, oe, hl} ->
          {pa, ba, oe, hl ++ [{:handle, event_type, pattern, body}]}

        other, acc ->
          # ignore anything else or handle
          acc
      end)

    {prompt_ast, backend_ast, on_enter_ast, handle_list}
  end

  defp parse_state_block(other), do: parse_state_block({:__block__, [], [other]})

  # We'll transform on_enter do ... end user code into a snippet returning a valid handle_event result
  defp expand_on_enter(block) do
    # user code might do "ask data, prompt" or "start_sse(data, prompt)" etc.
    block
  end

  # Turn user handle body into valid code. For example,
  # if user wrote:
  #   run_task extract_title(user_input)
  #   classify check_if_user_write_all(user_input) do ... end
  #
  # we need to transform those into calls to functions we generate or calls to EEx etc.
  defp generate_handle_body(body_ast, prompt_ast, backend_ast) do
    # For demonstration, let's just inline it.
    # You might do a macro expansion to interpret run_task, classify calls, etc.
    # If you want to run EEx for `prompt_ast`, you do that at runtime or compile time.
    # We'll keep it simple:
    body_ast
  end

  # -------------------------
  # tasks and classifications as helper functions
  # e.g. defp do_task_extract_title(args) do ... end
  # -------------------------
  defp build_tasks_ast(tasks) do
    # tasks is list of {task_name, arg_ast, block_ast}
    funs =
      for {task_name, arg_ast, block_ast} <- tasks do
        # parse block_ast to find prompt, backend
        {prompt, backend_mod} = parse_task_block(block_ast)
        # generate a function "defp do_task_<task_name>(args, data) do ... end"
        fun_name = String.to_atom("do_task_#{task_name}")

        quote do
          defp unquote(fun_name)(unquote(arg_ast), data) do
            # Use the prompt and backend to do something
            # Possibly EEx.eval_string
            prompt_str = unquote(prompt) || ""
            backend_mod = unquote(backend_mod) || Agent.API.Replicate
            # call your backend
            # e.g. backend_mod.call(prompt_str)
            {result, new_data} = {:fake_result, data}
            # Return updated data or something
            {result, new_data}
          end
        end
      end

    quote do
      (unquote_splicing(funs))
    end
  end

  defp parse_task_block({:__block__, _, lines}) do
    prompt_ast = nil
    backend_ast = nil

    {prompt_ast, backend_ast} =
      Enum.reduce(lines, {nil, nil}, fn
        {:prompt, p}, {pr, be} -> {p, be}
        {:backend, b}, {pr, be} -> {pr, b}
        _, acc -> acc
      end)

    {prompt_ast, backend_ast}
  end

  defp parse_task_block(other), do: parse_task_block({:__block__, [], [other]})

  # Same approach for classification blocks
  defp build_classifications_ast(classifications) do
    funs =
      for {class_name, arg_ast, block_ast} <- classifications do
        {prompt_ast, backend_ast} = parse_task_block(block_ast)
        fun_name = String.to_atom("do_classification_#{class_name}")

        quote do
          defp unquote(fun_name)(unquote(arg_ast), data) do
            prompt_str = unquote(prompt_ast) || ""
            backend_mod = unquote(backend_ast) || Agent.API.Replicate
            # do classification
            # e.g. result = backend_mod.call(prompt_str)
            {result, data}
          end
        end
      end

    quote do
      (unquote_splicing(funs))
    end
  end
end
