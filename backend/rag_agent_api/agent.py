"""
Google Gemini Agent module for the RAG Agent and API Layer system.

This module provides functionality for creating and managing a Google Gemini agent
that generates responses based on retrieved context.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .config import get_config
from .schemas import AgentContext, AgentResponse, SourceChunkSchema
from .utils import format_confidence_score


class GeminiAgent:
    """
    A class to manage the Google Gemini agent for generating responses based on context.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Google Gemini agent with configuration.

        Args:
            model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        """
        config = get_config()
        api_key = config.gemini_api_key

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Configure the Gemini client
        genai.configure(api_key=api_key)

        # Create the generative model instance
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.default_temperature = config.default_temperature

        logging.info(f"Gemini agent initialized with model: {model_name}")

    async def generate_response(self, context: AgentContext) -> AgentResponse:
        """
        Generate a response based on the provided context.

        Args:
            context: AgentContext containing the query and retrieved context chunks

        Returns:
            AgentResponse with the generated answer and metadata
        """
        # Check if retrieved context is empty (no chunks at all)
        if not context.retrieved_chunks:
            return AgentResponse(
                raw_response="I could not find this information in the book.",
                used_sources=[],
                confidence_score=0.0,
                is_valid=True,
                validation_details="No context chunks retrieved from the database",
                unsupported_claims=[]
            )

        # Check if context is insufficient (very short content)
        total_context_length = sum(len(chunk.content) for chunk in context.retrieved_chunks)
        if total_context_length < 10:  # Much lower threshold, but still meaningful
            return AgentResponse(
                raw_response="I could not find this information in the book.",
                used_sources=[],
                confidence_score=0.0,
                is_valid=True,
                validation_details="No sufficient context provided to answer the question",
                unsupported_claims=[]
            )

        try:
            # Prepare the system message with instructions for grounding responses
            system_message = self._create_system_message(context)

            # Prepare the user message with the query
            user_message = self._create_user_message(context)

            # For Google Gemini, we need to format the prompt differently
            # Combine system instructions and user query
            full_prompt = f"{system_message}\n\n{user_message}"

            # Generate response from Google Gemini
            # For async generation, we need to use the appropriate async method
            chat = self.model.start_chat()
            response = await chat.send_message_async(
                full_prompt,
                generation_config={
                    "temperature": context.source_policy if hasattr(context, 'temperature') else self.default_temperature,
                    "max_output_tokens": 1000
                }
            )

            # Extract the response text
            raw_response = response.text if response and hasattr(response, 'text') else str(response)

            # If the response indicates no information was found, return the exact message
            if "I could not find this information in the book" in raw_response:
                return AgentResponse(
                    raw_response="I could not find this information in the book.",
                    used_sources=[],
                    confidence_score=0.0,
                    is_valid=True,
                    validation_details="No relevant information found in the provided context",
                    unsupported_claims=[]
                )

            # Determine which sources were used (this is a simplified approach)
            used_sources = self._identify_used_sources(raw_response, context.retrieved_chunks)

            # Calculate confidence score (based on similarity scores of used sources)
            confidence_score = self._calculate_confidence_score(used_sources, context.retrieved_chunks)

            # Validate that the response is grounded in the provided context
            grounding_validation = self._validate_response_grounding(
                raw_response, context.retrieved_chunks, context.query
            )

            # Create and return the agent response
            agent_response = AgentResponse(
                raw_response=raw_response,
                used_sources=used_sources,
                confidence_score=confidence_score,
                is_valid=grounding_validation["is_valid"],
                validation_details=grounding_validation["details"],
                unsupported_claims=grounding_validation["unsupported_claims"]
            )

            logging.info(f"Agent response generated successfully. Confidence: {confidence_score:.2f}")
            return agent_response

        except Exception as e:
            logging.error(f"Error generating response from Google Gemini agent: {e}", exc_info=True)
            # Return the specific message when there's an error
            return AgentResponse(
                raw_response="I could not find this information in the book.",
                used_sources=[],
                confidence_score=0.0,
                is_valid=False,
                validation_details=f"Error generating response: {str(e)}",
                unsupported_claims=[]
            )

    def _create_system_message(self, context: AgentContext) -> str:
        """
        Create the system message that instructs the agent on how to behave.

        Args:
            context: AgentContext containing the query and retrieved context chunks

        Returns:
            Formatted system message string
        """
        system_prompt = """You are a documentation-based assistant.
Answer ONLY using the provided context from the book
"Physical AI & Humanoid Robotics".
If the answer is not found, reply EXACTLY:
"I could not find this information in the book."""
        return system_prompt

    def _create_user_message(self, context: AgentContext) -> str:
        """
        Create the user message containing the query.

        Args:
            context: AgentContext containing the query and retrieved context chunks

        Returns:
            Formatted user message string
        """
        return f"""CONTEXT:
{self._format_context_chunks(context.retrieved_chunks)}

QUESTION:
{context.query}"""

    def _format_context_chunks(self, chunks: List[SourceChunkSchema]) -> str:
        """
        Format the context chunks for the prompt.

        Args:
            chunks: List of source chunks to format

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append(f"[Chunk {i+1}]\n{chunk.content}\n[/Chunk {i+1}]")

        return "\n".join(formatted_chunks)

    def _create_context_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        """
        Create context messages from the retrieved chunks.
        With the new format, context is now provided in the user message,
        so this method returns an empty list to avoid duplication.

        Args:
            context: AgentContext containing the query and retrieved context chunks

        Returns:
            Empty list since context is now in user message
        """
        return []

    def _identify_used_sources(self, response: str, chunks: List[SourceChunkSchema]) -> List[str]:
        """
        Identify which sources were likely used in the response.
        This is a simplified approach - in a real implementation, you might use
        more sophisticated techniques like semantic similarity.

        Args:
            response: The agent's response text
            chunks: List of source chunks that were provided to the agent

        Returns:
            List of source IDs that were likely used
        """
        used_sources = []
        response_lower = response.lower()

        for chunk in chunks:
            # Check if any significant words from the chunk appear in the response
            content_words = set(chunk.content.lower().split()[:20])  # Check first 20 words
            response_words = set(response_lower.split())

            # If there's significant overlap, consider this chunk as used
            overlap = content_words.intersection(response_words)
            if len(overlap) > 2:  # Arbitrary threshold
                used_sources.append(chunk.id)

        # If no sources were identified, return all sources (conservative approach)
        if not used_sources:
            used_sources = [chunk.id for chunk in chunks]

        return used_sources

    def _calculate_confidence_score(self, used_sources: List[str], chunks: List[SourceChunkSchema]) -> float:
        """
        Calculate a confidence score based on the quality of the used sources.

        Args:
            used_sources: List of source IDs that were used
            chunks: List of all source chunks that were provided to the agent

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not used_sources:
            return 0.1  # Low confidence if no sources were used

        # Calculate average similarity score of used sources
        total_similarity = 0.0
        used_count = 0

        for chunk in chunks:
            if chunk.id in used_sources:
                total_similarity += chunk.similarity_score
                used_count += 1

        if used_count == 0:
            return 0.1  # Low confidence if no matching chunks found

        avg_similarity = total_similarity / used_count

        # If similarity scores are very low (e.g., due to embedding issues),
        # but we have content, still provide some confidence
        if avg_similarity < 0.1 and len(used_sources) > 0:
            # If we have relevant content but low similarity scores,
            # it might be due to embedding issues, not lack of relevance
            # So we'll set a minimum confidence if content exists
            return 0.3  # Low but not zero confidence
        else:
            # Normalize the confidence score (adjust based on your requirements)
            # Higher similarity scores contribute to higher confidence
            confidence = avg_similarity

        return format_confidence_score(confidence)

    def _validate_response_grounding(self, response: str, chunks: List[SourceChunkSchema], query: str) -> Dict[str, Any]:
        """
        Validate that the response is grounded in the provided context.

        Args:
            response: The agent's response text
            chunks: List of source chunks that were provided to the agent
            query: The original query

        Returns:
            Dictionary with validation results
        """
        # Check if the response contains elements from the provided context
        response_lower = response.lower()
        context_text = " ".join([chunk.content.lower() for chunk in chunks])

        # Simple heuristic: check if response contains significant terms from context
        response_words = set(response_lower.split())
        context_words = set(context_text.split())

        # Calculate overlap between response and context
        overlap = response_words.intersection(context_words)
        total_response_words = len(response_words)
        overlap_count = len(overlap)

        # If less than 30% of response words come from context, flag as potentially ungrounded
        is_grounded = True
        unsupported_claims = []

        if total_response_words > 0:
            grounding_ratio = overlap_count / total_response_words
            is_grounded = grounding_ratio >= 0.3  # At least 30% of words should come from context

        # For now, we'll just return the basic validation
        # In a more sophisticated implementation, you'd analyze the response more deeply
        details = f"Response grounding validation completed. Context overlap ratio: {overlap_count/total_response_words if total_response_words > 0 else 0:.2f}"

        return {
            "is_valid": is_grounded,
            "details": details,
            "unsupported_claims": unsupported_claims
        }

    async def validate_response_quality(self, response: str, context: AgentContext) -> bool:
        """
        Validate the quality of the agent's response.

        Args:
            response: The agent's response text
            context: AgentContext containing the query and retrieved context chunks

        Returns:
            True if response meets quality standards, False otherwise
        """
        # Check for common signs of poor quality responses
        if not response or response.strip() == "":
            logging.warning("Agent returned an empty response")
            return False

        # Check if response contains generic fallback phrases
        lower_response = response.lower()
        if "i don't know" in lower_response or "i don't have" in lower_response:
            # This might be a valid response if there's no relevant context
            if len(context.retrieved_chunks) == 0:
                return True  # Valid response if no context was provided
            else:
                # Check if the response is justified given the context
                # For now, we'll consider it valid if it acknowledges the lack of relevant information
                return True

        # In a more sophisticated implementation, you'd validate against the context more rigorously
        return True


# Global agent instance (if needed)
# agent_instance = OpenAIAgent()