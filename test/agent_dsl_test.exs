defmodule AgentDslTest do
  use ExUnit.Case
  doctest AgentDsl

  test "greets the world" do
    assert AgentDsl.hello() == :world
  end
end
