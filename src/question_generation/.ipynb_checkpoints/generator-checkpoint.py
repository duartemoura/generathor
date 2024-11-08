# src/question_generation/generator.py
from dataclasses import dataclass
from typing import List, Optional
import boto3
import json
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class QuestionAnswer:
    question: str
    answer: str
    context: Optional[str] = None

class EnhancedQuestionGenerator:
    """Enhanced question generator with context management"""

    def __init__(self,
                 llm_client=None,
                 model_id='anthropic.claude-v2',
                 embedding_manager=None,
                 max_tokens=500,
                 temperature=0.5):
        self.llm_client = llm_client or boto3.client('bedrock-runtime')
        self.model_id = model_id
        self.embedding_manager = embedding_manager
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_questions_from_docs(
            self,
            docs,
            num_questions: int = 10,
            questions_per_chunk: int = 2
    ) -> List[QuestionAnswer]:
        """
        Generate questions using semantic search for context
        """
        all_qa_pairs = []

        for doc in docs:
            # Use embedding-based search to find relevant chunks
            relevant_chunks = self.embedding_manager.find_relevant_chunks(
                doc.page_content,
                k=3
            )

            # Combine relevant chunks for context
            context = " ".join([c.page_content for c in relevant_chunks])

            # Generate questions for this context
            qa_pairs = self._generate_questions(
                context,
                questions_per_chunk
            )

            # Add context to QA pairs
            for qa in qa_pairs:
                qa.context = context

            all_qa_pairs.extend(qa_pairs)

            if len(all_qa_pairs) >= num_questions:
                break

        return all_qa_pairs[:num_questions]

    def _generate_questions(self, context: str, num_questions: int) -> List[dict]:
        """
        Generate question-answer pairs from the context
        """
        prompt = (
            f"You are a highly knowledgeable assistant trained to analyze technical and detailed content. "
            f"Based on the following text, generate {num_questions} high-quality, contextually relevant question-answer pairs. "
            f"Your task is to craft precise and well-structured questions that focus on the most critical aspects of the text, "
            f"and provide accurate, concise answers as if you have full understanding of the subject.\n\n"
            f"Text:\n{context}\n\n"
            "Guidelines:\n"
            "- The questions should capture the core ideas, technical nuances, and details in the text.\n"
            "- Avoid general or trivial questions; focus on the key points or concepts that require understanding.\n"
            "- The answers should be clear, accurate, and directly address the questions without introducing extraneous information.\n"
            "- Use formal language and maintain technical accuracy.\n\n"
            "Provide the output in this format:\n"
            "Question 1: [Precise question text]\n"
            "Answer 1: [Clear and accurate answer text]\n"
            "Question 2: [Precise question text]\n"
            "Answer 2: [Clear and accurate answer text]\n"
            "...\n\n"
            "Start your response below:\n"
        )

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.llm_client.invoke_model(
            modelId=self.model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )

        response_body = response['body'].read().decode('utf-8')
        response_json = json.loads(response_body)

        # Extract the generated text from content
        if 'content' in response_json and isinstance(response_json['content'], list):
            generated_text = ''.join(item['text'] for item in response_json['content'] if item['type'] == 'text')
        else:
            logger.error(f"Unexpected response structure: {response_json}")
            raise KeyError("Could not find generated text in response")

        qa_pairs = self._parse_qna_pairs(generated_text)
        return qa_pairs


    def _parse_qna_pairs(self, generated_text: str) -> List[QuestionAnswer]:
        """
        Parses the generated text to extract question and answer pairs.
        """
        qa_pairs = []
        # Regex pattern to match 'Question x: ... Answer x: ...'
        pattern = r"Question\s*\d+\s*:\s*(.+?)\s*Answer\s*\d+\s*:\s*(.+?)(?=Question\s*\d+\s*:|$)"
        matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
        for question, answer in matches:
            qa_pairs.append(QuestionAnswer(
                question=question.strip(),
                answer=answer.strip()
            ))
        return qa_pairs
