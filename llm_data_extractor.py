from openai import OpenAI
import base64
import os


class LLMDataExtractor:
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_message(
        self,
        input_text: str,
        image_path: str = None,
        message_type: str = "analysis",
        plot_num: int = None,
    ) -> list:
        """
        Create different message structures based on input type.

        Args:
            input_text: The text content to analyze
            message_type: Type of message to create ('analysis', 'extraction', 'summary', 'classification')

        Returns:
            List of message dictionaries for OpenAI API
        """

        if image_path:
            # Encode image to base64 if provided
            image_base64 = self._encode_image_to_base64(image_path)

        if message_type == "ThT_plot_identifier":
            messages = [
                {
                    "role": "system",
                    "content": """You are given a scientific figure where each plot is labeled with a unique number and enclosed within a bounding box.

        Your task is to identify which of these plots are Thioflavin T (ThT) fluorescence vs. time plots. Only consider plots that have bounding boxes, and only include those that are line or scatter plots (not bar charts, images, or tables).

        Return ONLY a Python-style list of the relevant plot numbers. For example: [1, 3, 4]. If there are none, return an empty list: []. Do not include any additional explanation or text.""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ]

        elif message_type == "data_extractor" and plot_num:
            user_content = [
                {
                    "type": "text",
                    "text": input_text,  # Should include caption + surrounding context
                }
            ]

            # Add image if provided
            if image_path:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "auto",
                        },
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": f"""You are provided with a scientific figure and its associated caption and text. Plot {plot_num} has been identified as a Thioflavin T (ThT) fluorescence vs. time plot.

        Your task is to extract the experimental conditions specifically associated with plot {plot_num}.

        Return a JSON object with exactly this structure:

        {{
        "{plot_num}": {{
            "Protein": ["string"],
            "Mutation": ["string or null"],
            "Protein concentration": ["number with unit"],
            "Temperature": ["number with unit"],
            "pH": [number],
            "Additives": ["string"],
            "Additive concentrations": ["string with unit"]
        }}
        }}

        Each list must have the same number of items as the number of variables/lines in the plot. For example, if the plot has 3 lines, each list must have 3 items ([a,b,c]), even if the condition is the same across all lines.

        If a condition has multiple values, they should be separated by commas. For example if there are two additives, it should be ["additive1, additive2"], and the additive concentrations should match.
        
        If a condition is not mentioned, use an empty list.
        """,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

        elif message_type == "data_extractor_2" and plot_num:
            user_content = [
                {
                    "type": "text",
                    "text": input_text,  # Should include caption + surrounding context
                }
            ]

            # Add image if provided
            if image_path:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "auto",
                        },
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": """You are provided with a scientific figure and its associated caption and text. Plot {plot_num} has been identified as a Thioflavin T (ThT) fluorescence vs. time plot.

        Your task is to extract the experimental conditions specifically associated with plot {plot_num}. Extract the mapping between each legend key or line style (e.g., •, ○, "black triangle", "dashed red line") and its corresponding experimental condition.

        Return a JSON object where each key is a legend key, and the value is a dictionary of experimental variables.

        Use this format:
        {
        "•": {
            "Protein": "AChE586-599",
            "Protein concentration": "100 µM",
            "Mutation": "WT",
            "Temperature": null,
            "pH": null,
            "Additives": ["ThT"],
            "Additive concentrations": ["165 µM"]
        },
        ...
        }

        If there is more than one additive in the list, make sure additive concentrations match.

        """,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

        elif message_type == "classification":
            messages = [
                {
                    "role": "system",
                    "content": """You are a scientific literature classifier. Classify research content based on experimental approaches, 
                    research domains, and methodological categories. Provide clear classifications with confidence levels.""",
                },
                {
                    "role": "user",
                    "content": f"""Classify this scientific text according to:
                    1. Research domain (e.g., biochemistry, biophysics, cell biology)
                    2. Experimental approach (e.g., in vitro, in vivo, computational)
                    3. Protein aggregation relevance (high/medium/low)
                    4. Key methodologies used
                    
                    Text to classify:
                    {input_text}""",
                },
            ]

        else:
            # Default generic message
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful scientific research assistant. Analyze the provided text and provide relevant insights.",
                },
                {"role": "user", "content": input_text},
            ]

        return messages

    def use_message(self, messages: list, model: str = "gpt-4o-mini", **kwargs) -> dict:
        """
        Send messages to OpenAI API and get response.

        Args:
            messages: List of message dictionaries
            model: OpenAI model to use (default: gpt-4o-mini)
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary containing the response and metadata
        """

        # Default parameters
        default_params = {
            "temperature": 0.1,
            "max_tokens": 1500,  # Increased from 500 to handle complex legend mappings
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Update with any provided kwargs
        params = {**default_params, **kwargs}

        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, **params
            )

            result = {
                "success": True,
                "content": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
                "error": None,
            }

        except Exception as e:
            result = {
                "success": False,
                "content": None,
                "model": model,
                "usage": None,
                "finish_reason": None,
                "error": str(e),
            }

        return result

    def run_model(
        self,
        text: str,
        image_path: str = None,
        analysis_type: str = "data_extractor",
        model: str = "gpt-4o-mini",
        tht_plot_list: list = None,
        **kwargs,
    ) -> dict:
        """
        Convenience method to create message and get response in one call.

        Args:
            text: Text to analyze
            image_path: Path to image file
            analysis_type: Type of analysis ('analysis', 'extraction', 'summary', 'classification')
            model: OpenAI model to use
            tht_plot_list: List of ThT plot numbers for data extraction
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary containing the response and metadata
        """
        # For data_extractor, we need to pass the first plot number from the list
        plot_num = (
            tht_plot_list[0] if tht_plot_list and len(tht_plot_list) > 0 else None
        )
        messages = self.create_message(text, image_path, analysis_type, plot_num)
        return self.use_message(messages, model, **kwargs)
