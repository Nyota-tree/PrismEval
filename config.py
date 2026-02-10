"""
Configuration: prompt templates, model settings, and batch parameters.

Customize here (or via configs/default.yaml):
- API provider and model selection
- System / user / evaluation prompt templates
- Batch concurrency and retry settings
- CSV column names
"""

# API configuration
API_PROVIDER = "deepseek"

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 2000

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
ANTHROPIC_TEMPERATURE = 0.7
ANTHROPIC_MAX_TOKENS = 2000

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"
DEEPSEEK_REASONER_TEMPERATURE = 0.0
DEEPSEEK_REASONER_TOP_P = 1.0
DEEPSEEK_REASONER_MAX_TOKENS = 2000

# Batch configuration
MAX_WORKERS = 5
MAX_RETRIES = 3
RETRY_DELAY = 1

# Batch generator columns
INPUT_COLUMN = "input_text"
OUTPUT_COLUMN = "model_response"

SYSTEM_PROMPT = """You are a helpful AI assistant.

# Task
Your task is to process the given input and generate a response according to the requirements.

# Instructions
1. Analyze the input carefully
2. Generate a response that meets the requirements
3. Follow the output format specified below

# Output Format
Please output your response in the following XML structure:

<thinking>
...your thinking process...
</thinking>

<final_result>
    <priority>P{x}</priority>
    <content>
    ...your generated content...
    </content>
</final_result>

Note: You can customize this prompt template in config.py to match your specific use case."""

USER_PROMPT_TEMPLATE = "{input_text}"

# Batch evaluator columns
MODEL_RESPONSE_COLUMN = "model_response"
FINAL_CONTENT_COLUMN = "final_content"
INPUT_TEXT_COLUMN = "input_text"

EVALUATOR_API_PROVIDER = "deepseek"
EVALUATOR_MODEL = "deepseek-reasoner"

EVALUATION_PROMPT = """You are an expert evaluator. Your task is to evaluate the quality of AI-generated content.

# Task
You will receive two pieces of text:
1. **Original Input**: The original input text.
2. **Model Output**: The AI-generated output content.

Please evaluate the model output based on your criteria and return a JSON object.

# Inputs
<original_input>
{original_text}
</original_input>

<model_output>
{model_output}
</model_output>

# Evaluation Criteria
Please evaluate from the following dimensions (0-10 points each):

1. **Factuality** (weight 30%): Accuracy of facts, no hallucinations
2. **Completeness** (weight 20%): Coverage of key information
3. **Adherence** (weight 25%): Following instructions and format requirements
4. **Quality** (weight 25%): Overall quality and readability

# Output Format
Return ONLY a JSON object (no Markdown code blocks):

{{
  "determined_priority": "P0/P1/P2/P3",
  "scores": {{
    "factuality_score": <0-10>,
    "completeness_score": <0-10>,
    "adherence_score": <0-10>,
    "attractiveness_score": <0-10>
  }},
  "weighted_total_score": <calculated_total_0_to_100>,
  "reasoning": "Brief explanation of your evaluation",
  "pass": <true/false>
}}

Note:
- Each dimension score is 0-10 points
- weighted_total_score = (factuality_score * 3) + (completeness_score * 2) + (adherence_score * 2.5) + (attractiveness_score * 2.5)
- weighted_total_score ranges from 0 to 100 (weights sum to 10)
- If factuality_score < 5, pass must be false
- Decision threshold: weighted_total_score >= 75 for PUBLISH

You can customize this evaluation prompt in config.py to match your specific evaluation criteria."""

EVAL_COLUMN_PREFIX = "score_"
REASONING_COLUMN_NAME = "eval_reasoning"
