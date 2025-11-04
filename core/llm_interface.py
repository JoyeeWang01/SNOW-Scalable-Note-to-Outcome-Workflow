"""
LLM interface functions for querying different API providers.
"""

import json
import time
import requests
from typing import Any, Dict, List, Callable
from dataclasses import dataclass


def load_api_config(provider: str):
    """
    Load API configuration for the specified provider.

    Args:
        provider: One of "gemini", "claude", or "openai"

    Returns:
        Module containing the API configuration

    Raises:
        ValueError: If provider is not recognized
    """
    provider = provider.lower()

    if provider == "gemini":
        from config import api_gemini as api_config
    elif provider == "claude":
        from config import api_claude as api_config
    elif provider == "openai":
        from config import api_openai as api_config
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: gemini, claude, openai")

    return api_config

from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
    Part,
    ResponseValidationError
)

from core.log_utils import print


def query_llm(prompt, api_config, thinking=False, return_full_response=False):
    """
    Query an LLM with the given prompt using the specified API provider.

    Args:
        prompt: The prompt to send to the LLM
        api_config: API configuration module (e.g., from load_api_config())
        thinking: Whether to enable thinking mode
        return_full_response: If True, returns dict with 'text', 'thinking', and 'raw' keys

    Returns:
        str or dict: The LLM response text, or full response dict if return_full_response=True
    """

    api_provider = api_config.llm_provider
    model = api_config.model
    url = getattr(api_config, 'url', None)
    key = getattr(api_config, 'key', None)

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Content-Type': 'application/json',
    }

    if api_provider == "claude":
        if thinking:
            payload = json.dumps({
                "model_id": model,
                "prompt_text": prompt,
                "max_tokens": 20000,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            })
        else:
            payload = json.dumps({
                "model_id": model,
                "prompt_text": prompt,
                "max_tokens": 20000,
                "thinking": {
                    "type": "disabled"
                }
            })

    attempt = 1
    while True:
        try:
            if api_provider == "claude":
                response = requests.request("POST", url, headers=headers, data=payload, timeout=600)
                response.raise_for_status()
                response_json = response.json()

                if return_full_response:
                    # Return full response with thinking trace if available
                    result = {
                        'raw': response_json,
                        'thinking': None,
                        'text': None
                    }
                    if thinking and len(response_json.get("content", [])) > 1:
                        result['thinking'] = response_json["content"][0].get("thinking", "")
                        result['text'] = response_json["content"][1]["text"]
                    else:
                        result['text'] = response_json["content"][0]["text"]
                    return result
                else:
                    # Return just the text (backward compatible)
                    if thinking:
                        return response_json["content"][1]["text"]
                    else:
                        return response_json["content"][0]["text"]
            elif api_provider == "openai":
                openai_client = api_config.client
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
                text_response = response.choices[0].message.content

                if return_full_response:
                    # Return full response dict (OpenAI doesn't have thinking traces)
                    return {
                        'raw': response,
                        'thinking': None,
                        'text': text_response
                    }
                else:
                    return text_response
            elif api_provider == "gemini":
                # Generate content
                if thinking:
                    response = api_config.gemini_model_thinking.generate_content(prompt)
                else:
                    response = api_config.gemini_model.generate_content(prompt)

                # Extract text from response
                text_response = response.candidates[0].content.parts[0].text

                if return_full_response:
                    # Return full response dict (Gemini doesn't have thinking traces)
                    return {
                        'raw': response,
                        'thinking': None,
                        'text': text_response
                    }
                else:
                    return text_response
            else:
                raise ValueError(f"Unsupported api_provider: {api_provider}")

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            print(f"Request failed (attempt {attempt}), retrying in 60 seconds...")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Details: {e}")
            time.sleep(60)
            attempt += 1
        except KeyError as e:
            print(f"KeyError: {e}. Response structure may be different. Attempt {attempt}")
            print(f"Response content: {response.json() if 'response' in locals() else 'No response'}")
            time.sleep(60)
            attempt += 1
        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt}): {e}")
            print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
            time.sleep(60)
            attempt += 1
        except Exception as e:
            print(f"Unexpected error (attempt {attempt}): {type(e).__name__}: {e}")
            print(f"API provider: {api_provider}")
            import traceback
            traceback.print_exc()
            time.sleep(60)
            attempt += 1


@dataclass
class ToolDefinition:
    """Defines a tool that the LLM can call"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class LLMToolCaller:
    """
    A class for calling LLMs with tool support across different providers.
    """

    def __init__(self, api_url: str, subscription_key: str, model: str, tool_list: List[ToolDefinition],
                 api_provider: str = "claude", timeout: int = 600, gemini_model=None):
        self.api_url = api_url
        self.subscription_key = subscription_key
        self.model = model
        self.tools = {}
        self.timeout = timeout
        self.api_provider = api_provider
        self.thinking = False

        # Gemini-specific setup
        if api_provider == "gemini":
            # Convert tool definitions to Gemini format
            self.gemini_tools = self._convert_to_gemini_tools(tool_list)

            # Initialize Gemini models (flash for standard, pro for thinking)
            self.gemini_model_flash = GenerativeModel(
                "gemini-2.5-flash",
                tools=self.gemini_tools
            )
            self.gemini_model_pro = GenerativeModel(
                "gemini-2.5-pro",
                tools=self.gemini_tools
            )

            # Store tools for later execution
            for tool_def in tool_list:
                self.add_tool(tool_def)
        else:
            # Claude/OpenAI setup
            self.tool_prompt = self.format_tool_definitions_to_tools(tool_list)
            for tool_def in tool_list:
                self.add_tool(tool_def)

    def add_tool(self, tool: ToolDefinition):
        """Add a tool that the LLM can call"""
        self.tools[tool.name] = tool

    def _convert_to_gemini_tools(self, tool_list: List[ToolDefinition]) -> List[Tool]:
        """Convert ToolDefinition objects to Gemini Tool format"""
        function_declarations = []
        for tool_def in tool_list:
            # Convert parameters to Gemini format
            gemini_func = FunctionDeclaration(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters
            )
            function_declarations.append(gemini_func)

        # Return a single Tool containing all function declarations
        return [Tool(function_declarations=function_declarations)]

    def format_tool_definitions_to_tools(self, tool_list: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert ToolDefinition objects to the tools format expected by the API"""
        tool_prompt = []
        for tool_def in tool_list:
            self.add_tool(tool_def)
            # Process parameters dictionary into properties format
            properties = {}
            required = []

            for param_name, param_config in tool_def.parameters.items():
                if isinstance(param_config, dict):
                    # If param_config is already a schema dict, use it directly
                    properties[param_name] = param_config
                    # Check if this parameter is marked as required
                    required.append(param_name)

            tool = {
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            tool_prompt.append(tool)
        return tool_prompt

    def _make_api_call(self, prompt: str) -> str:
        """Make the API call to the LLM"""
        if self.api_provider == "claude":
            if not self.tools:
                if self.thinking:
                    payload = json.dumps({
                        "model_id": self.model,
                        "prompt_text": prompt,
                        "max_tokens": 50000,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 30000
                        }
                    })
                else:
                    payload = json.dumps({
                        "model_id": self.model,
                        "prompt_text": prompt,
                        "max_tokens": 20000,
                        "thinking": {
                            "type": "disabled"
                        }
                    })
            else:
                if self.thinking:
                    payload = json.dumps({
                        "model_id": self.model,
                        "prompt_text": prompt,
                        "max_tokens": 50000,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 30000
                        },
                        "tools": self.tool_prompt
                    })
                else:
                    payload = json.dumps({
                        "model_id": self.model,
                        "prompt_text": prompt,
                        "max_tokens": 20000,
                        "thinking": {
                            "type": "disabled"
                        },
                        "tools": self.tool_prompt
                    })
        elif self.api_provider == "openai":
            if not self.tools:
                payload = json.dumps({
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            else:
                payload = json.dumps({
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "tools": self.tool_prompt
                })

        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'application/json'
        }

        attempt = 1
        while True:
            try:
                response = requests.request("POST", self.api_url, headers=headers, data=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt}, retrying...")
                time.sleep(60)
                attempt += 1
            except requests.exceptions.RequestException as e:
                print(f"Request failed on attempt {attempt}, retrying...")
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Details: {e}")
                time.sleep(60)
                attempt += 1
            except KeyError as e:
                print(f"KeyError: {e}. Response structure may be different. Attempt {attempt}")
                print(f"Response content: {response.json() if 'response' in locals() else 'No response'}")
                time.sleep(60)
                attempt += 1
            except json.JSONDecodeError as e:
                print(f"JSON decode error (attempt {attempt}): {e}")
                print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
                time.sleep(60)
                attempt += 1
            except Exception as e:
                print(f"Unexpected error (attempt {attempt}): {e}")
                time.sleep(60)
                attempt += 1

    def _extract_tool_calls(self, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text content and tool calls from the API response"""
        content = api_response.get('content', [])

        tool_calls = []

        for item in content:
            if item.get('type') == 'tool_use':
                tool_calls.append({
                    'name': item.get('name'),
                    'id': item.get('id'),
                    'parameters': item.get('input', {})
                })

        return tool_calls

    def _extract_text(self, api_response: Dict[str, Any]) -> str:
        """Extract text content from the API response"""
        content = api_response.get('content', [])

        for item in content:
            if item.get('type') == 'text':
                return item["text"]

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call"""
        tool_name = tool_call.get('name')
        parameters = tool_call.get('parameters', {})
        tool_id = tool_call.get('id')

        if tool_name not in self.tools:
            error_result = {
                'type': 'tool_result',
                'tool_use_id': tool_id,
                'content': f"Error: Tool '{tool_name}' not found",
                'is_error': True
            }
            return error_result

        try:
            tool = self.tools[tool_name]
            result = tool.function(**parameters)

            return {
                'type': 'tool_result',
                'tool_use_id': tool_id,
                'content': str(result),
                'is_error': False
            }
        except Exception as e:
            error_result = {
                'type': 'tool_result',
                'tool_use_id': tool_id,
                'content': f"Error executing {tool_name}: {str(e)}",
                'is_error': True
            }
            return error_result

    def _query_gemini_with_tools(self, user_prompt: str, additional_prompt: str = "", thinking: bool = False, max_iterations: int = 100, feature_name: str = None, iteration: int = None) -> str:
        """
        Query Gemini with tool support
        """
        # Build detailed log for conversation
        conversation_log_parts = []
        conversation_log_parts.append("=" * 80)
        conversation_log_parts.append("INITIAL USER PROMPT")
        conversation_log_parts.append("=" * 80)
        conversation_log_parts.append(user_prompt)
        conversation_log_parts.append("")

        # Select model based on thinking mode
        gemini_model = self.gemini_model_pro if thinking else self.gemini_model_flash
        model_name = "gemini-2.5-pro" if thinking else "gemini-2.5-flash"
        print(f"Using {model_name} for this query")

        # Start a chat session
        chat = gemini_model.start_chat()

        while True:
            # Send the initial message
            try:
                response = chat.send_message(user_prompt)
                break
            except ResponseValidationError as e:
                print(f"Caught a malformed response from the model: {e}")

        # Process tool calls if any
        for iteration in range(max_iterations):

            # Check if the response contains function calls
            if response.candidates and response.candidates[0].function_calls:
                # Log tool calls
                conversation_log_parts.append("=" * 80)
                conversation_log_parts.append(f"TOOL CALLS (Iteration {iteration + 1})")
                conversation_log_parts.append("=" * 80)

                # Collect all function responses before sending back to Gemini
                function_response_parts = []

                # Execute each function call
                tool_call_idx = 0
                for function_call in response.candidates[0].function_calls:
                    tool_call_idx += 1
                    function_name = function_call.name

                    # Convert args to dict (Gemini uses a different format)
                    args = {}
                    for key, value in function_call.args.items():
                        # Handle different value types
                        if hasattr(value, 'string_value'):
                            args[key] = value.string_value
                        elif hasattr(value, 'number_value'):
                            args[key] = value.number_value
                        elif hasattr(value, 'bool_value'):
                            args[key] = value.bool_value
                        else:
                            args[key] = str(value)

                    print(f"Executing {function_name} with args: {args}")

                    # Log this tool call
                    conversation_log_parts.append(f"\nTool Call {tool_call_idx}:")
                    conversation_log_parts.append(f"  Name: {function_name}")
                    conversation_log_parts.append(f"  Arguments: {args}")

                    # Execute the function
                    if function_name in self.tools:
                        try:
                            tool = self.tools[function_name]
                            result = tool.function(**args)

                            # Log result in conversation
                            conversation_log_parts.append(f"  Result: {str(result)}")

                            # Add function response to the list
                            function_response_parts.append(
                                Part.from_function_response(
                                    name=function_name,
                                    response={"content": str(result)}
                                )
                            )
                        except Exception as e:
                            print(f"Error executing {function_name}: {e}")

                            # Log error in conversation
                            conversation_log_parts.append(f"  Error: {str(e)}")

                            # Add error response to the list
                            function_response_parts.append(
                                Part.from_function_response(
                                    name=function_name,
                                    response={"error": str(e)}
                                )
                            )
                    else:
                        print(f"Function {function_name} not found in tools")

                        # Log error in conversation
                        conversation_log_parts.append(f"  Error: Function {function_name} not found")

                        # Add not found response
                        function_response_parts.append(
                            Part.from_function_response(
                                name=function_name,
                                response={"error": f"Function {function_name} not found"}
                            )
                        )

                # Send all function responses back to Gemini in a single message
                if function_response_parts:
                    conversation_log_parts.append("")
                    conversation_log_parts.append("Sending tool results back to LLM...")
                    conversation_log_parts.append("")

                    while True:
                        try:
                            response = chat.send_message(Content(parts=function_response_parts))
                            break
                        except ResponseValidationError as e:
                            print(f"Caught a malformed response from the model: {e}")
            else:
                # No more function calls, return the text response
                conversation_log_parts.append("=" * 80)
                conversation_log_parts.append("FINAL RESPONSE (No more tool calls)")
                conversation_log_parts.append("=" * 80)

                # Handle multiple content parts case
                final_text = ""
                try:
                    # First try the simple response.text
                    final_text = response.text
                except ValueError as e:
                    # If we get "Multiple content parts are not supported", extract from parts
                    if "Multiple content parts" in str(e):
                        print("Handling multiple content parts response...")
                        # Try to get text from the parts directly
                        if response.candidates and response.candidates[0].content.parts:
                            # Usually the JSON response is in the second part (index 1)
                            for i, part in enumerate(response.candidates[0].content.parts):
                                if hasattr(part, 'text') and part.text:
                                    print(f"Found text in part {i} (length: {len(part.text)})")
                                    # Check if this looks like the JSON response we want
                                    if '<JSON>' in part.text or '"decision"' in part.text:
                                        print(f"Returning JSON response from part {i}")
                                        final_text = part.text
                                        break
                            # If we didn't find JSON, return the last text part
                            if not final_text:
                                for part in reversed(response.candidates[0].content.parts):
                                    if hasattr(part, 'text') and part.text:
                                        final_text = part.text
                                        break
                    if not final_text:
                        raise  # Re-raise if it's a different error

                # Log final response
                conversation_log_parts.append(final_text)

                # Log complete conversation
                try:
                    from core.log_utils import log_tool_conversation
                    log_tool_conversation("\n".join(conversation_log_parts), feature_name, iteration)
                except Exception as log_error:
                    print(f"Warning: Failed to log tool conversation: {log_error}")

                return final_text

    def query_with_tools(self, user_prompt: str, additional_prompt: str = "", thinking: bool = False, max_iterations: int = 100, feature_name: str = None, iteration: int = None) -> Dict[str, Any]:
        """
        Query the LLM with tool support

        Args:
            user_prompt: The user's input prompt
            additional_prompt: Additional context to include with tool results
            thinking: Enable thinking mode
            max_iterations: Maximum number of tool call iterations to prevent infinite loops
            feature_name: Optional feature name for logging context
            iteration: Optional iteration number for logging context

        Returns:
            Dict containing the conversation history and final response
        """
        # Handle Gemini separately
        if self.api_provider == "gemini":
            return self._query_gemini_with_tools(user_prompt, additional_prompt, thinking, max_iterations, feature_name, iteration)

        # Original logic for Claude/OpenAI
        self.thinking = thinking
        conversation_history = []
        conversation_history.append({
            'role': 'user',
            'content': user_prompt
        })

        # Build detailed log for conversation
        conversation_log_parts = []
        conversation_log_parts.append("=" * 80)
        conversation_log_parts.append("INITIAL USER PROMPT")
        conversation_log_parts.append("=" * 80)
        conversation_log_parts.append(user_prompt)
        conversation_log_parts.append("")

        for tool_round in range(max_iterations):
            # Get LLM response
            llm_response = self._make_api_call(str(conversation_history))
            conversation_history.append({
                'role': 'assistant',
                'content': llm_response['content']
            })

            # Log assistant response
            conversation_log_parts.append("=" * 80)
            conversation_log_parts.append(f"ASSISTANT RESPONSE (Tool Round {tool_round + 1})")
            conversation_log_parts.append("=" * 80)
            conversation_log_parts.append(str(llm_response['content']))
            conversation_log_parts.append("")

            # Check for tool calls
            tool_calls = self._extract_tool_calls(llm_response)

            if not tool_calls:
                # No tool calls, we're done
                conversation_log_parts.append("=" * 80)
                conversation_log_parts.append("FINAL RESPONSE (No more tool calls)")
                conversation_log_parts.append("=" * 80)

                final_text = ""
                if self.thinking:
                    final_text = llm_response['content'][1]["text"]
                else:
                    final_text = llm_response['content'][0]["text"]

                conversation_log_parts.append(final_text)

                # Log complete conversation
                try:
                    from core.log_utils import log_tool_conversation
                    log_tool_conversation("\n".join(conversation_log_parts), feature_name, iteration)
                except Exception as log_error:
                    print(f"Warning: Failed to log tool conversation: {log_error}")

                return final_text

            # Log tool calls
            conversation_log_parts.append("=" * 80)
            conversation_log_parts.append(f"TOOL CALLS (Tool Round {tool_round + 1})")
            conversation_log_parts.append("=" * 80)
            for i, tool_call in enumerate(tool_calls):
                conversation_log_parts.append(f"\nTool Call {i + 1}:")
                conversation_log_parts.append(f"  Name: {tool_call.get('name')}")
                conversation_log_parts.append(f"  Parameters: {tool_call.get('parameters')}")

            # Execute tool calls
            tool_results = [{'type': 'text', 'content': additional_prompt}]
            conversation_log_parts.append("")
            conversation_log_parts.append("=" * 80)
            conversation_log_parts.append(f"TOOL RESULTS (Tool Round {tool_round + 1})")
            conversation_log_parts.append("=" * 80)

            for i, tool_call in enumerate(tool_calls):
                result = self._execute_tool_call(tool_call)
                tool_results.append(result)

                conversation_log_parts.append(f"\nTool Result {i + 1}:")
                conversation_log_parts.append(f"  Tool: {tool_call.get('name')}")
                conversation_log_parts.append(f"  Result: {result.get('content')}")
                conversation_log_parts.append(f"  Error: {result.get('is_error', False)}")

            conversation_log_parts.append("")

            conversation_history.append({
                'role': 'user',
                'content': tool_results,
            })

        # Max iterations reached
        conversation_log_parts.append("=" * 80)
        conversation_log_parts.append("MAX ITERATIONS REACHED")
        conversation_log_parts.append("=" * 80)

        # Log complete conversation
        try:
            from core.log_utils import log_tool_conversation
            log_tool_conversation("\n".join(conversation_log_parts), feature_name, iteration)
        except Exception as log_error:
            print(f"Warning: Failed to log tool conversation: {log_error}")

        return {
            'final_response': "Maximum iterations reached",
            'conversation_history': conversation_history,
            'tool_calls_made': sum(1 for entry in conversation_history if entry['type'] == 'tool_results'),
            'iterations': max_iterations,
            'warning': 'Maximum iterations reached - conversation may be incomplete'
        }
