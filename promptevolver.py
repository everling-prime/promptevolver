#!/usr/bin/env python3
"""
PromptEvolver: Automatic prompt optimization using reasoning models and promptfoo.

A Python CLI tool that runs promptfoo tests, uses a reasoning model to analyze failures,
generates improved prompts, and repeats until optimized.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import yaml
import requests


@dataclass
class TestResult:
    """Single test result from promptfoo"""
    vars: Dict[str, Any]
    prompt: str
    output: str
    score: float
    passed: bool
    reason: Optional[str] = None


@dataclass
class EvaluationRound:
    """One complete iteration of evaluation"""
    iteration: int
    prompt_variant: str
    pass_rate: float
    total_score: float
    failed_tests: List[TestResult]
    reasoning_analysis: str
    improvement_suggestions: List[str]


class PromptEvolver:
    """Main class for prompt evolution optimization."""
    
    def __init__(
        self,
        config_path: str,
        reasoning_model: str,
        max_iterations: int,
        use_openai_nano: bool = False,
    ):
        """Initialize the PromptEvolver.
        
        Args:
            config_path: Path to promptfoo config YAML file
            reasoning_model: Ollama model name for reasoning
            max_iterations: Maximum number of optimization iterations
        """
        self.config_path = Path(config_path)
        self.use_openai_nano = use_openai_nano
        self.reasoning_model = reasoning_model
        self.max_iterations = max_iterations
        self.evolution_history: List[EvaluationRound] = []
        
        # Load initial configuration
        self.base_prompts: List[str] = []
        self._load_config()
        
        # Determine .env location (same directory as this script)
        self.env_file = Path(__file__).resolve().parent / '.env'
        if self.env_file.exists():
            self.promptfoo_env_args = ['--env-file', str(self.env_file)]
            self._load_env_file()
        else:
            self.promptfoo_env_args = []
            print(f"⚠️  .env not found at {self.env_file}. Falling back to existing environment variables.")

        self.ollama_base_url = "http://localhost:11434"
        self.reasoning_available = False
        if self.use_openai_nano:
            self.reasoning_model = "gpt-5-nano"
            if not os.getenv("OPENAI_API_KEY"):
                print("⚠️  OPENAI_API_KEY not set. Reasoning steps will be skipped.")
            else:
                self.reasoning_available = True
                print(f"🤖 Using OpenAI model: {self.reasoning_model}")
        else:
            # Initialize Ollama client
            try:
                self._test_ollama_connection()
                self.reasoning_available = True
            except Exception as e:
                print(f"⚠️  Ollama not available: {e}. Reasoning steps will be skipped.")
        
    def _load_config(self) -> None:
        """Load promptfoo configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            prompts_raw = self.config.get('prompts', [])
            if not prompts_raw:
                raise ValueError("Config contains no prompts")
            self.base_prompts = [self._extract_prompt_text(p) for p in prompts_raw]
            if not self.base_prompts:
                raise ValueError("Unable to extract prompt text from configuration")
            # Work with plain string prompts going forward
            self.config['prompts'] = list(self.base_prompts)
            self.initial_prompt = self.base_prompts[0]
            print(f"📋 Loaded config from {self.config_path}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")

    @staticmethod
    def _extract_prompt_text(prompt_entry: Any) -> str:
        """Normalize prompt definitions to their text form."""
        if isinstance(prompt_entry, dict):
            for key in ('text', 'prompt', 'raw'):
                value = prompt_entry.get(key)
                if isinstance(value, str):
                    return value
            for value in prompt_entry.values():
                if isinstance(value, str):
                    return value
            return ''
        return str(prompt_entry or '')

    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama and check if model is available."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check if model exists (with or without tag)
            model_found = False
            matched_name = None
            for name in model_names:
                # Match exact name or name without tag
                if name == self.reasoning_model or name.startswith(f"{self.reasoning_model}:"):
                    model_found = True
                    matched_name = name
                    break
            
            if not model_found:
                available_models = ', '.join(model_names)
                raise ValueError(f"Model '{self.reasoning_model}' not found. Available models: {available_models}")
            
            # Use the matched name (with tag)
            self.reasoning_model = matched_name
            print(f"🤖 Connected to Ollama with model: {self.reasoning_model}")
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama connection failed: {e}")
    
    def _call_ollama(self, prompt: str, clean_thinking=True) -> str:
        """Make a call to Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.reasoning_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            result = data.get('response', '').strip()
            
            if not result:
                raise RuntimeError("Ollama returned empty response")
            
            # Remove thinking tags if requested
            if clean_thinking:
                # Remove everything between </think> and </think> tags
                result = re.sub(r'</think>.*?</think>', '', result, flags=re.DOTALL)
                result = result.strip()
            
            if not result:
                raise RuntimeError("Ollama response became empty after cleaning")
            
            return result
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

    def _load_env_file(self) -> None:
        """Load environment variables from the colocated .env file."""
        try:
            with open(self.env_file, 'r') as env_fp:
                for raw_line in env_fp:
                    line = raw_line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.lower().startswith('export '):
                        line = line[7:].strip()
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as exc:
            print(f"⚠️  Failed to load environment variables from {self.env_file}: {exc}")

    def _call_openai_model(self, prompt: str, clean_thinking: bool = True) -> str:
        """Call OpenAI chat completions API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.reasoning_model,
            "input": prompt,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            text_chunks: List[str] = []

            output_text = data.get("output_text")
            if isinstance(output_text, list):
                text_chunks.extend(str(chunk) for chunk in output_text if chunk)
            elif isinstance(output_text, str) and output_text:
                text_chunks.append(output_text)

            if not text_chunks and isinstance(data.get("output"), list):
                for item in data["output"]:
                    if isinstance(item, dict):
                        maybe_text = item.get("text")
                        if maybe_text:
                            text_chunks.append(str(maybe_text))
                        if isinstance(item.get("content"), list):
                            for part in item["content"]:
                                if isinstance(part, dict):
                                    part_text = part.get("text")
                                    if part_text:
                                        text_chunks.append(str(part_text))
                                elif isinstance(part, str):
                                    text_chunks.append(part)

            if not text_chunks:
                raise RuntimeError("OpenAI API returned empty content")

            result_text = "\n".join(chunk.strip() for chunk in text_chunks if chunk).strip()

            if clean_thinking and '<think>' in result_text:
                result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()

            return result_text
        except Exception as e:
            if isinstance(e, requests.HTTPError) and e.response is not None:
                raise RuntimeError(
                    f"OpenAI API call failed: {e}. Response: {e.response.text}"
                ) from e
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _call_reasoning_model(self, prompt: str, clean_thinking=True) -> str:
        """Dispatch reasoning call to the selected provider."""
        if self.use_openai_nano:
            return self._call_openai_model(prompt, clean_thinking=clean_thinking)
        return self._call_ollama(prompt, clean_thinking=clean_thinking)

    def _reasoning_unavailable_message(self) -> str:
        if self.use_openai_nano:
            return "OpenAI gpt-5-nano unavailable; skipping analysis."
        return "Ollama not available; skipping analysis."

    def run_promptfoo_eval(self) -> Dict[str, Any]:
        """Execute promptfoo evaluation and return raw results.
        
        Returns:
            Raw results dictionary from promptfoo
        """
        print("🔬 Running promptfoo evaluation...")
        
        try:
            # Run promptfoo evaluation
            command = ['npx', 'promptfoo@latest', 'eval', '-o', 'output/latest.json']
            if self.promptfoo_env_args:
                command.extend(self.promptfoo_env_args)

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.config_path.parent
            )
            
            if result.returncode not in [0, 100]:  # 100 means some tests failed but evaluation completed
                print(f"❌ Promptfoo error: {result.stderr}")
                raise RuntimeError(f"Promptfoo evaluation failed: {result.stderr}")
            
            # Load results from output file
            output_file = self.config_path.parent / 'output' / 'latest.json'
            if not output_file.exists():
                raise RuntimeError("Promptfoo output file not found")
            
            with open(output_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            raise RuntimeError(f"Failed to run promptfoo evaluation: {e}")
    
    def parse_results(self, raw_results: Dict[str, Any]) -> Tuple[List[TestResult], float, float]:
        """Parse promptfoo results into structured format.
        
        Args:
            raw_results: Raw JSON results from promptfoo
            
        Returns:
            Tuple of (test_results, pass_rate, average_score)
        """
        test_results = []
        total_score = 0.0
        passed_count = 0
        
        # Extract results from promptfoo structure
        results_table = raw_results.get('results', {}).get('results', [])
        
        for row in results_table:
            vars_data = row.get('vars', {})
            prompt_data = row.get('prompt', {})
            prompt = prompt_data.get('raw', '')
            response_data = row.get('response', {})
            output_text = response_data.get('output', '')
            grading = row.get('gradingResult', {})
            
            score = grading.get('score', 0.0)
            passed = grading.get('pass', False)
            reason = grading.get('reason', '')
            
            test_result = TestResult(
                vars=vars_data,
                prompt=prompt,
                output=output_text,
                score=score,
                passed=passed,
                reason=reason
            )
            
            test_results.append(test_result)
            total_score += score
            if passed:
                passed_count += 1
        
        # Calculate metrics
        pass_rate = passed_count / len(test_results) if test_results else 0.0
        avg_score = total_score / len(test_results) if test_results else 0.0
        
        return test_results, pass_rate, avg_score
    
    def analyze_failures_with_reasoning(self, failed_tests: List[TestResult]) -> Tuple[str, List[str]]:
        """Analyze failed tests using the configured reasoning model.
        
        Args:
            failed_tests: List of failed test results
            
        Returns:
            Tuple of (reasoning_analysis, improvement_suggestions)
        """
        if not failed_tests:
            return "No failures to analyze", []

        if not self.reasoning_available:
            return self._reasoning_unavailable_message(), []

        relevant_failures = failed_tests[:5]

        failure_summary = []
        for i, test in enumerate(relevant_failures, 1):
            failure_summary.append(
                f"Test {i}:\nVariables: {test.vars}\nOutput: {test.output[:300]}...\nScore: {test.score}\nReason: {test.reason}\n"
            )

        failures_text = "\n".join(failure_summary)

        # Determine context from variables to provide appropriate analysis
        context_type = "general"
        if failed_tests and "{{code}}" in str(failed_tests[0].vars):
            context_type = "code review"
        elif failed_tests and ("{{message}}" in str(failed_tests[0].vars) or 
                              any("customer" in str(test.vars).lower() or 
                                  "inquiry" in str(test.vars).lower() or 
                                  "refund" in str(test.vars).lower() 
                                  for test in failed_tests)):
            context_type = "customer support"

        if context_type == "customer support":
            analysis_prompt = f"""Analyze these customer support test failures and suggest prompt improvements.

FAILED TESTS:
{failures_text}

The prompt should help customer service agents respond better to customer inquiries. Focus on empathy, clear solutions, and professional tone.

Output ONLY valid JSON:
{{
  "reasoning": "Brief analysis of why customer responses are failing",
  "suggestions": [
    "Suggestion 1: specific improvement for customer support",
    "Suggestion 2: specific improvement for customer support", 
    "Suggestion 3: specific improvement for customer support"
  ]
}}"""
        elif context_type == "code review":
            analysis_prompt = f"""Analyze these code review test failures and suggest prompt improvements.

FAILED TESTS:
{failures_text}

The prompt should help reviewers provide better code feedback with concrete examples and actionable suggestions.

Output ONLY valid JSON:
{{
  "reasoning": "Brief analysis of why code reviews are failing",
  "suggestions": [
    "Suggestion 1: specific improvement for code review",
    "Suggestion 2: specific improvement for code review",
    "Suggestion 3: specific improvement for code review"
  ]
}}"""
        else:
            analysis_prompt = f"""Analyze these test failures and suggest prompt improvements.

FAILED TESTS:
{failures_text}

Identify what's wrong and suggest 3-5 specific improvements.

Output ONLY valid JSON:
{{
  "reasoning": "Brief analysis of the pattern of failures",
  "suggestions": [
    "Suggestion 1: specific improvement",
    "Suggestion 2: specific improvement",
    "Suggestion 3: specific improvement"
  ]
}}"""

        try:
            response_text = self._call_reasoning_model(analysis_prompt)
            
            if not response_text:
                return "Empty response from reasoning model", ["Review failed tests manually"]
            
            # Extract JSON from code blocks if present
            json_text = response_text
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    json_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end != -1:
                    json_text = response_text[start:end].strip()
            
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as je:
                print(f"⚠️  JSON parsing failed: {je}")
                # Try to extract suggestions using regex as fallback
                try:
                    import re
                    suggestions = []
                    reasoning = "Analysis completed but JSON parsing failed"
                    
                    # Extract reasoning
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', json_text)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1)
                    
                    # Extract suggestions
                    suggestion_matches = re.findall(r'"([^"]+)"', json_text)
                    for match in suggestion_matches:
                        if 'Suggestion' in match or len(match) > 20:
                            suggestions.append(match)
                    
                    if suggestions:
                        return reasoning, suggestions[:5]
                except:
                    pass
                
                return f"JSON parsing failed: {je}", ["Review failed tests manually"]
            
            reasoning = data.get('reasoning', 'Analysis failed')
            suggestions = data.get('suggestions', [])
            
            # Ensure suggestions is a list of strings
            if isinstance(suggestions, str):
                suggestions = [suggestions]
            elif not isinstance(suggestions, list):
                suggestions = []
            
            return reasoning, suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            print(f"⚠️  Reasoning analysis failed: {e}")
            return f"Analysis failed: {e}", ["Review failed tests manually"]
    
    def improve_prompt(self, current_prompt: str, suggestions: List[str]) -> str:
        """Generate improved prompt using the configured reasoning model."""
        if not suggestions:
            return current_prompt
            
        suggestions_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

        # Determine context for improvement prompt
        if "{{message}}" in current_prompt or any("customer" in s.lower() or "inquiry" in s.lower() for s in suggestions_text):
            context_type = "customer support"
            good_example = '"Respond to this customer inquiry: {{message}}. Show empathy, offer a clear solution, and maintain a professional tone."'
        else:
            context_type = "general"
            good_example = '"Review this code: {{{{code}}}}. Identify bugs, suggest improvements with examples, and explain your reasoning."'

        improvement_prompt = f"""Make this prompt SIMPLER and more effective.

CURRENT PROMPT:
{current_prompt}

ISSUES TO FIX:
{suggestions_text}

CRITICAL RULES:
- Make it SHORTER and SIMPLER, not longer or more complex
- Keep variable placeholders like {{{{code}}}} or {{{{message}}}}
- Be direct and conversational, not formal or prescriptive
- Focus on ONE clear improvement at a time
- Remove any unnecessary instructions or complexity
- Maximum 2-3 sentences
- Avoid colon-separated labels; prefer full sentences instead

GOOD EXAMPLE:
{good_example}

BAD EXAMPLE:
Long multi-paragraph instructions with detailed validation rules and complex formatting requirements.

NEW SIMPLIFIED PROMPT:"""

        if not self.reasoning_available:
            print(f"⚠️  {self._reasoning_unavailable_message()}")
            return current_prompt
        try:
            response_text = self._call_reasoning_model(improvement_prompt, clean_thinking=True)
            
            # Extract just the prompt if model added extra text
            # First, try to extract quoted text
            import re
            quoted_match = re.search(r'"([^"]+{{[^"]+}}[^"]*)"', response_text)
            if quoted_match:
                response_text = quoted_match.group(1)
            else:
                # Look for the actual prompt after markers
                lines = response_text.split('\n')
                prompt_lines = []
                found_prompt = False
                
                for line in lines:
                    stripped = line.strip()
                    # Skip conversational wrappers
                    if stripped.lower().startswith(('sure', 'here', 'okay', 'alright', 'let me')):
                        continue
                    # Skip empty lines and common markers
                    if not stripped or stripped.lower().startswith(('new', 'prompt:', 'simplified', 'improved')):
                        if stripped:
                            found_prompt = True
                        continue
                    # Stop at explanatory text
                    if found_prompt and (stripped.startswith('-') or stripped.startswith('*') or 
                                        stripped.lower().startswith(('this', 'the', 'note', 'explanation', 'i removed', 'let me know'))):
                        break
                    prompt_lines.append(stripped)
                
                # Use extracted prompt or full response
                if prompt_lines:
                    response_text = ' '.join(prompt_lines).strip()
            
            # Validate the new prompt
            if not response_text or len(response_text) < 20 or '{{' not in response_text:
                print(f"⚠️  Got unexpected response, keeping current prompt")
                return current_prompt
            
            # Reject if absurdly long (absolute limit)
            if len(response_text) > 2000:
                print(f"⚠️  New prompt too long ({len(response_text)} chars), keeping current")
                return current_prompt
            
            # Allow reasonable growth - be more permissive to enable real improvements
            if len(current_prompt) > 500:
                max_length = len(current_prompt) * 1.5  # 1.5x growth for very long prompts
            elif len(current_prompt) > 200:
                max_length = len(current_prompt) * 4  # 4x growth for medium prompts
            else:
                max_length = len(current_prompt) * 10  # 10x growth for short prompts
            
            # But always cap at reasonable absolute maximum
            max_length = min(max_length, 1200)
            
            if len(response_text) > max_length:
                print(f"⚠️  New prompt too complex ({len(response_text)} chars vs {max_length} max), keeping current")
                return current_prompt
            
            return response_text.strip()
            
        except Exception as e:
            print(f"⚠️  Prompt improvement failed: {e}")
            return current_prompt
    
    def update_config_with_new_prompt(self, new_prompt: str) -> None:
        """Update configuration with new prompt.
        
        Args:
            new_prompt: New prompt text to use
        """
        self.config['prompts'] = [new_prompt]
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to update config: {e}")
    
    def run_iteration(self, iteration: int, current_prompt: str) -> EvaluationRound:
        """Run a single optimization iteration.
        
        Args:
            iteration: Iteration number
            current_prompt: Current prompt to test
            
        Returns:
            EvaluationRound with results
        """
        print(f"\n{'='*60}")
        print(f"🔄 Iteration {iteration}")
        print(f"{'='*60}")
        
        # Update config with current prompt
        self.update_config_with_new_prompt(current_prompt)
        
        # Run evaluation
        raw_results = self.run_promptfoo_eval()
        test_results, pass_rate, avg_score = self.parse_results(raw_results)
        
        # Display results
        print(f"📊 Results: Pass Rate: {pass_rate:.1%}, Avg Score: {avg_score:.2f}")
        
        # Analyze failures
        failed_tests = [t for t in test_results if not t.passed]
        reasoning_analysis = ""
        suggestions = []
        
        if failed_tests:
            print(f"❌ {len(failed_tests)} tests failed")
            reasoning_analysis, suggestions = self.analyze_failures_with_reasoning(failed_tests)
            
            print(f"\n🧠 Analysis:")
            print(reasoning_analysis)
            
            print(f"\n✨ Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("🎉 All tests passed!")
            reasoning_analysis = "All tests passed successfully"
            suggestions = []
        
        return EvaluationRound(
            iteration=iteration,
            prompt_variant=current_prompt,
            pass_rate=pass_rate,
            total_score=avg_score,
            failed_tests=failed_tests,
            reasoning_analysis=reasoning_analysis,
            improvement_suggestions=suggestions
        )
    
    def evolve(self) -> None:
        """Run the complete optimization process."""
        print("\n🚀 Starting PromptEvolver Optimization")
        print(f"📋 Config: {self.config_path}")
        print(f"🧠 Model: {self.reasoning_model}")
        print(f"🔄 Max Iterations: {self.max_iterations}")
        print(f"📝 Initial Prompt: {self.initial_prompt[:100]}...")
        
        current_prompt = self.initial_prompt
        best_prompt = current_prompt
        best_score = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            # Run iteration
            round_result = self.run_iteration(iteration, current_prompt)
            self.evolution_history.append(round_result)
            
            # Track best performing prompt
            if round_result.total_score > best_score:
                best_score = round_result.total_score
                best_prompt = current_prompt
                print(f"✨ New best score: {best_score:.2f}")
            
            # Check if we're done
            if round_result.pass_rate >= 0.95:
                print(f"\n🎯 Target pass rate achieved! ({round_result.pass_rate:.1%})")
                break
            
            # Generate improved prompt if we have failed tests and haven't reached target
            if round_result.failed_tests and round_result.pass_rate < 0.95:
                print(f"\n🔧 Generating improved prompt...")
                new_prompt = self.improve_prompt(current_prompt, round_result.improvement_suggestions)
                
                # Only use new prompt if it's actually different
                if new_prompt != current_prompt:
                    print(f"\n📝 New prompt preview:")
                    print(f"{'─'*40}")
                    print(new_prompt[:200] + "..." if len(new_prompt) > 200 else new_prompt)
                    print(f"{'─'*40}")
                    
                    # Test new prompt in next iteration
                    # If it performs worse, we'll revert to best_prompt
                    current_prompt = new_prompt
                else:
                    print(f"\n⚠️  No improvement generated, keeping current prompt")
                
                # Small delay to avoid rate limits
                time.sleep(1)
        
        # If final prompt performed worse than best, note it
        if self.evolution_history and self.evolution_history[-1].total_score < best_score:
            print(f"\n📊 Note: Best performing prompt was from an earlier iteration (score: {best_score:.2f})")
            print(f"   Consider using: python promptevolver.py --view-iteration {self.evolution_history.index([r for r in self.evolution_history if r.prompt_variant == best_prompt][0]) + 1}")
        
        # Print final summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        # Reset promptfooconfig.yaml to base prompts for the next run
        self._reset_config_to_base()
        
    def print_summary(self) -> None:
        """Print optimization summary."""
        print(f"\n{'='*60}")
        print("📈 OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        if not self.evolution_history:
            print("No iterations completed")
            return
        
        initial_round = self.evolution_history[0]
        final_round = self.evolution_history[-1]
        
        print(f"Iterations completed: {len(self.evolution_history)}")
        print(f"Initial pass rate: {initial_round.pass_rate:.1%}")
        print(f"Final pass rate: {final_round.pass_rate:.1%}")
        
        improvement = ((final_round.pass_rate - initial_round.pass_rate) / 
                      initial_round.pass_rate * 100) if initial_round.pass_rate > 0 else 0
        print(f"Improvement: {improvement:+.1f}%")
        
        print(f"\n📊 Iteration-by-iteration results:")
        for round_result in self.evolution_history:
            print(f"  Iteration {round_result.iteration}: "
                  f"{round_result.pass_rate:.1%} pass rate, "
                  f"{round_result.total_score:.2f} avg score")
        
        print(f"\n🎯 Final optimized prompt:")
        print(f"{'─'*40}")
        print(final_round.prompt_variant)
        print(f"{'─'*40}")
    
    def save_results(self) -> None:
        """Save optimization results to JSON file."""
        results = {
            'config_path': str(self.config_path),
            'reasoning_model': self.reasoning_model,
            'max_iterations': self.max_iterations,
            'initial_prompt': self.initial_prompt,
            'evolution_history': [asdict(round_result) for round_result in self.evolution_history],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('evolution_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to evolution_results.json")
    
    def _reset_config_to_base(self) -> None:
        """Reset the prompt configuration file back to the original base prompts."""
        try:
            # Load current config from disk to preserve unrelated keys/structure
            with open(self.config_path, 'r') as f:
                current_cfg = yaml.safe_load(f) or {}
            current_cfg['prompts'] = list(self.base_prompts)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(
                    current_cfg,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            print("\n🔄 Reset promptfooconfig.yaml to base prompts for next run")
        except Exception as e:
            print(f"\n⚠️  Failed to reset config to base prompts: {e}")
        
        # Save promptfoo iterations
        self._save_promptfoo_iterations()
    
    def _save_promptfoo_iterations(self) -> None:
        """Save each iteration as a separate promptfoo config."""
        iterations_dir = self.config_path.parent / 'iterations'
        iterations_dir.mkdir(exist_ok=True)
        
        for round_result in self.evolution_history:
            iteration_config = self.config.copy()
            iteration_config['prompts'] = [round_result.prompt_variant]
            
            iteration_file = iterations_dir / f'iteration_{round_result.iteration}.yaml'
            with open(iteration_file, 'w') as f:
                yaml.safe_dump(
                    iteration_config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        
        print(f"📁 Iterations saved to {iterations_dir}/")
        print(f"\n📖 To view iterations with promptfoo:")
        print(f"   python promptevolver.py --view-iteration 1")
        print(f"   python promptevolver.py --view-iteration {len(self.evolution_history)}")
        print(f"   (Viewer opens in browser and runs in background)")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='PromptEvolver - Automatic prompt optimization')
    parser.add_argument('--config', default='promptfooconfig.yaml', 
                       help='Path to promptfoo config file')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Maximum number of optimization iterations')
    parser.add_argument('--model', default='qwen3:0.6b',
                       help='Reasoning model name for Ollama (ignored with --use-openai-nano)')
    parser.add_argument('--view-iteration', type=int,
                       help='Open promptfoo viewer for specific iteration')
    parser.add_argument('--compare', action='store_true',
                       help='Compare iterations in promptfoo viewer')
    parser.add_argument('--use-openai-nano', action='store_true',
                        help='Use OpenAI gpt-5-nano for analysis and prompt revision')
    
    args = parser.parse_args()
    
    try:
        if args.view_iteration:
            # View specific iteration
            iteration_file = Path(f'iterations/iteration_{args.view_iteration}.yaml')
            if not iteration_file.exists():
                print(f"❌ Iteration file not found: {iteration_file}")
                sys.exit(1)
            
            print(f"📖 Opening iteration {args.view_iteration} in promptfoo viewer...")
            subprocess.Popen(['npx', 'promptfoo@latest', 'view', str(iteration_file.parent), '-y'])
            print(f"✅ Browser opened with promptfoo viewer running in background")
            
        elif args.compare:
            # Open promptfoo viewer with latest iteration
            iterations_dir = Path('iterations')
            if not iterations_dir.exists():
                print("❌ No iterations directory found")
                sys.exit(1)
            
            # Find the latest iteration file
            iteration_files = list(iterations_dir.glob('iteration_*.yaml'))
            if not iteration_files:
                print("❌ No iteration files found")
                sys.exit(1)
            
            latest_file = max(iteration_files, key=lambda f: int(f.stem.split('_')[1]))
            print(f"📊 Opening comparison view with {latest_file.name}...")
            subprocess.Popen(['npx', 'promptfoo@latest', 'view', str(latest_file.parent), '-y'])
            print(f"✅ Browser opened with promptfoo comparison view running in background")
            
        else:
            # Run optimization
            if args.use_openai_nano:
                args.model = 'gpt-5-nano'
            evolver = PromptEvolver(
                args.config,
                args.model,
                args.iterations,
                use_openai_nano=args.use_openai_nano,
            )
            evolver.evolve()
            
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
