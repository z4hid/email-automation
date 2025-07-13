"""
Email Automation Engine using DSPy (Declarative Self-improving Python)

This module provides an AI-powered email processing system that:
1. Classifies incoming emails into business categories
2. Generates professional draft replies automatically
3. Uses Google's Gemini AI model through DSPy framework

Main Components:
- EmailClassifierSignature: Defines how to classify emails
- EmailReplySignature: Defines how to generate replies  
- EmailProcessor: The main DSPy module that combines classification + reply generation
- EmailEngine: High-level interface for the entire system

DSPy Framework Benefits:
- Declarative programming instead of manual prompt engineering
- Automatic optimization of prompts and examples
- Better reliability than traditional prompt-based approaches
- Built-in few-shot learning capabilities
"""

import dspy
import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
import threading
from typing import Dict, List, Optional, Union

# Load environment variables from .env file
load_dotenv()

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailClassifierSignature(dspy.Signature):
    """
    DSPy Signature for Email Classification
    
    A signature in DSPy defines the input and output structure for an AI task.
    This signature tells the AI: "Given email text, classify it into business categories"
    
    Categories we classify emails into:
    - Quote Request: Customer asking for pricing/quotes
    - New Order Received: Customer placing an order  
    - Delivery Follow-up: Customer asking about delivery status
    - Other: Everything else that doesn't fit above categories
    """
    
    # Input field: The complete email content to analyze
    email_text = dspy.InputField(
        desc="The full content of the email including subject, sender, and body text."
    )
    
    # Output field: The classification result
    category = dspy.OutputField(
        desc="The most appropriate business category: Quote Request, New Order Received, Delivery Follow-up, or Other."
    )


class EmailReplySignature(dspy.Signature):
    """
    DSPy Signature for Email Reply Generation
    
    This signature tells the AI: "Given an email category and content, write a professional reply"
    
    The AI will generate context-appropriate responses:
    - Quote Requests → Acknowledge request, ask for details, mention follow-up timeline
    - New Orders → Confirm receipt, provide order details, mention processing timeline  
    - Delivery Follow-ups → Provide status update, estimated delivery time
    - Other → Polite acknowledgment and appropriate next steps
    """
    
    # Input fields: What the AI needs to generate a good reply
    email_category = dspy.InputField(
        desc="The business category of the email (Quote Request, New Order Received, etc.)"
    )
    email_content = dspy.InputField(
        desc="The original email content that we're replying to"
    )
    
    # Output field: The generated reply
    draft_reply = dspy.OutputField(
        desc="A professional, helpful email reply appropriate for the category and content"
    )


class EmailProcessor(dspy.Module):
    """
    Main DSPy Module for Email Processing
    
    A DSPy Module is like a class that combines multiple AI operations.
    This module does two things in sequence:
    1. Classify the email into a business category
    2. Generate an appropriate reply based on that category
    
    DSPy Modules are composable - you can chain them together,
    optimize them with different algorithms, and reuse them.
    """
    
    def __init__(self):
        """Initialize the email processor with two DSPy components"""
        super().__init__()
        
        # Predict: Simple one-shot prediction (good for classification)
        # We use this for email classification because it's a straightforward categorization task
        self.classifier = dspy.Predict(EmailClassifierSignature)
        
        # ChainOfThought: Multi-step reasoning (good for complex generation)
        # We use this for reply generation because writing good emails requires reasoning
        # about context, tone, and appropriate responses
        self.reply_generator = dspy.ChainOfThought(EmailReplySignature)
    
    def forward(self, email_text: str) -> dspy.Prediction:
        """
        Process an email through classification and reply generation
        
        Args:
            email_text: The complete email content to process
            
        Returns:
            dspy.Prediction containing:
            - category: The classified email category
            - draft_reply: The generated reply text
        """
        
        # Step 1: Classify the email
        # This calls the Gemini AI model with our classification signature
        classification_result = self.classifier(email_text=email_text)
        
        # Step 2: Generate reply only for business categories
        # We skip generating replies for "Other" category to avoid unnecessary responses
        if classification_result.category != "Other":
            # This calls Gemini AI again with our reply generation signature
            # ChainOfThought will show step-by-step reasoning in the response
            reply_result = self.reply_generator(
                email_category=classification_result.category, 
                email_content=email_text
            )
            
            return dspy.Prediction(
                category=classification_result.category, 
                draft_reply=reply_result.draft_reply
            )
        
        # For "Other" category, return a standard message
        return dspy.Prediction(
            category=classification_result.category, 
            draft_reply="Thank you for your email. We have received your message and will respond appropriately."
        )


class EmailEngine:
    """
    High-Level Email Processing Engine
    
    This is the main interface that other parts of the application use.
    It handles:
    - Setting up the DSPy framework with Gemini AI
    - Managing the EmailProcessor module
    - Providing training examples for better performance
    - Thread-safe operations for web applications
    
    Design Philosophy:
    - Simple interface for complex AI operations
    - Robust error handling and logging
    - Thread-safe for concurrent email processing
    - Lazy initialization for better startup performance
    """
    
    def __init__(self):
        """Initialize the engine (but don't set up AI yet for faster startup)"""
        
        # Core components (initialized later)
        self.lm = None  # Language Model (Gemini AI)
        self.processor = None  # Our EmailProcessor module
        self.is_initialized = False  # Track initialization status
        
        # Thread safety for web applications
        # Multiple emails might be processed simultaneously
        self._lock = threading.Lock()
        
        logger.info("📧 EmailEngine created (not initialized yet)")
        
    def initialize(self, api_key: Optional[str] = None) -> bool:
        """
        Initialize the DSPy framework and AI models
        
        This is separated from __init__ because:
        - AI model setup takes time and network calls
        - We want fast application startup
        - We can retry initialization if it fails
        - Better error handling for missing API keys
        
        Args:
            api_key: Optional Gemini API key (will use environment variable if not provided)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        
        # Thread safety: Only one thread should initialize at a time
        with self._lock:
            # Don't initialize twice
            if self.is_initialized:
                logger.info("✅ Engine already initialized")
                return True
            
            try:
                # Step 1: Get API key from parameter or environment
                if not api_key:
                    api_key = os.getenv("GEMINI_API_KEY")
                
                if not api_key:
                    logger.error("❌ GEMINI_API_KEY not found")
                    logger.error("Please add GEMINI_API_KEY=your_key_here to your .env file")
                    logger.error("Get your key from: https://makersuite.google.com/app/apikey")
                    return False
                
                # Step 2: Initialize Gemini AI through DSPy
                logger.info(f"🔑 Found API Key: {api_key[:10]}{'*' * (len(api_key) - 10)}")
                logger.info("🤖 Initializing Gemini AI model...")
                
                # DSPy LM (Language Model) setup
                # This creates a connection to Google's Gemini Flash model
                self.lm = dspy.LM(
                    model="gemini/gemini-2.5-flash",  # Fast, cost-effective Gemini model
                    api_key=api_key
                )
                
                # Configure DSPy to use our language model globally
                # This tells all DSPy operations to use Gemini AI
                dspy.configure(lm=self.lm)
                
                # Step 3: Create and optimize our email processor
                logger.info("⚙️ Setting up email processor with training examples...")
                self.processor = self._create_optimized_processor()
                
                # Step 4: Mark as successfully initialized
                self.is_initialized = True
                logger.info("✅ EmailEngine initialized successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize EmailEngine: {e}")
                logger.error("Check your API key and internet connection")
                self.is_initialized = False
                return False
    
    def _get_training_examples(self) -> List[dspy.Example]:
        """
        Create training examples for better AI performance
        
        DSPy uses "few-shot learning" - giving the AI examples of good
        classifications helps it perform better on new emails.
        
        These examples are real-world business email patterns that teach the AI:
        - What language indicates a quote request
        - How to recognize new orders  
        - What delivery follow-up emails look like
        - Professional response patterns for each category
        
        Returns:
            List of DSPy Example objects for training
        """
        
        training_examples = [
            # Example 1: Quote Request Pattern
            dspy.Example(
                email_text=(
                    "Subject: RFQ - Costing for Custom Sensor Assemblies\n\n"
                    "Hello Sales Team,\n\n"
                    "We require a formal RFQ for a new project. Could you please provide "
                    "costing for the custom sensor assemblies outlined in the attached "
                    "drawings (QD-DWG-77A and QD-DWG-77B)?\n\n"
                    "Please price for a batch of 50 and a batch of 100.\n\n"
                    "Best regards,\nJohn Smith"
                ),
                category="Quote Request"
            ).with_inputs("email_text"),  # Tell DSPy this is an input example

            # Example 2: Another Quote Request Pattern
            dspy.Example(
                email_text=(
                    "Subject: Request for Quote - Replacement Motor Looms\n\n"
                    "Hi Team,\n\n"
                    "Could you please provide a quote for the following replacement "
                    "parts for our weaving machines?\n\n"
                    "ITEM: AI-ML-V4, Industrial Motor Looms\n"
                    "QUANTITY: 10 units\n\n"
                    "Thank you"
                ),
                category="Quote Request"
            ).with_inputs("email_text"),

            # Example 3: New Order Pattern
            dspy.Example(
                email_text=(
                    "Subject: Purchase Order PO2025-095 for Sensor Assemblies\n\n"
                    "Hi Jennifer,\n\n"
                    "Thank you for the quick turnaround on the quote.\n\n"
                    "Please see the attached Purchase Order PO2025-095 for the batch "
                    "of 100 units. This order is based on your quotation Q-9981.\n\n"
                    "Best regards"
                ),
                category="New Order Received"
            ).with_inputs("email_text"),

            # Example 4: Delivery Follow-up Pattern
            dspy.Example(
                email_text=(
                    "Subject: Delivery Inquiry for PO-PW-1134\n\n"
                    "Hello Brian,\n\n"
                    "I'm following up on our order for motor looms, PO-PW-1134. "
                    "Can you please provide an estimated delivery date? We need to "
                    "schedule technicians for the installation and need to know the "
                    "expected arrival at our facility.\n\n"
                    "Thanks"
                ),
                category="Delivery Follow-up"
            ).with_inputs("email_text"),

            # Example 5: Quote Refresh Pattern
            dspy.Example(
                email_text=(
                    "Subject: Quote Refresh - Part SC-3100-D Power Converters\n\n"
                    "Hi Jennifer,\n\n"
                    "Could you please provide a refreshed quote for 200 units of "
                    "part SC-3100-D Power Converters? Our last PO was in January, "
                    "and we just need to verify the current costing before issuing "
                    "a new order.\n\n"
                    "Thank you"
                ),
                category="Quote Request"
            ).with_inputs("email_text"),

            # Example 6: Another New Order Pattern
            dspy.Example(
                email_text=(
                    "Subject: New PO Attached - PO-2025-790 for SC-3100-D\n\n"
                    "Hi Jennifer,\n\n"
                    "Thanks for sending that over.\n\n"
                    "Please find our new PO attached for the power converters. "
                    "The order attached, PO-2025-790, is for 200 units.\n\n"
                    "Best regards"
                ),
                category="New Order Received"
            ).with_inputs("email_text"),

            # Example 7: Delivery ETD Request
            dspy.Example(
                email_text=(
                    "Subject: ETD Request for PO2025-095\n\n"
                    "Hi Jennifer,\n\n"
                    "Could you provide the target ETD for our order PO2025-095? "
                    "Our logistics team is planning the receiving schedule and needs "
                    "to know the estimated date of dispatch from your facility.\n\n"
                    "Thanks"
                ),
                category="Delivery Follow-up"
            ).with_inputs("email_text"),

            # Example 8: RFQ with Reference
            dspy.Example(
                email_text=(
                    "Subject: RFQ for Sub-Component Machining - Ref: Our Assemblies SC-8900\n\n"
                    "Hi Aisha,\n\n"
                    "Hope you're having a good week.\n\n"
                    "We are sending an RFQ for a new project. Per our conversation, "
                    "please see attached drawings for a machined sub-component. We need "
                    "pricing for these parts which will be used in our SC-8900 assemblies.\n\n"
                    "Best regards"
                ),
                category="Quote Request"
            ).with_inputs("email_text"),

            # Example 9: Delivery Confirmation
            dspy.Example(
                email_text=(
                    "Subject: Confirmation for order PO-2025-790\n\n"
                    "Hi Jennifer,\n\n"
                    "Just a quick email to confirm that order PO-2025-790 is on track "
                    "to ship out this Friday as per the acknowledged delivery date.\n\n"
                    "Please let me know if there are any delays.\n\n"
                    "Thanks"
                ),
                category="Delivery Follow-up"
            ).with_inputs("email_text"),

            # Example 10: Additional Order
            dspy.Example(
                email_text=(
                    "Subject: Purchase Order for additional Motor Looms - PO-PW-1145\n\n"
                    "Hello,\n\n"
                    "Please accept the attached Purchase Order for an additional 5 units "
                    "of the AI-ML-V4 Motor Looms. This is a follow-up to our last PO "
                    "(PO-PW-1134).\n\n"
                    "Best regards"
                ),
                category="New Order Received"
            ).with_inputs("email_text")
        ]
        
        logger.info(f"📚 Created {len(training_examples)} training examples for AI optimization")
        return training_examples
    
    def _create_optimized_processor(self) -> EmailProcessor:
        """
        Create and optimize the EmailProcessor using DSPy's training capabilities
        
        DSPy offers several optimization strategies:
        - LabeledFewShot: Uses our training examples to improve performance
        - BootstrapFewShot: Automatically generates more examples
        - MIPROv2: Advanced optimization (requires more compute)
        
        We use LabeledFewShot because:
        - Fast setup and execution
        - Works well with our manually curated examples
        - Good balance of performance and simplicity
        - Reliable for production use
        
        Returns:
            Optimized EmailProcessor ready for email processing
        """
        
        try:
            # Get our training examples
            training_examples = self._get_training_examples()
            
            # Create the basic email processor
            email_processor = EmailProcessor()
            
            # Use DSPy's LabeledFewShot optimizer
            # k=3 means use 3 most relevant examples for each prediction
            # This gives the AI context without overwhelming it
            optimizer = dspy.teleprompt.LabeledFewShot(k=3)
            
            logger.info("🚀 Optimizing email processor with DSPy (this improves AI performance)...")
            
            # Compile/optimize the processor with our examples
            # This process teaches the AI our business patterns
            optimized_processor = optimizer.compile(
                student=email_processor,  # The module to optimize
                trainset=training_examples  # Our training examples
            )
            
            logger.info("✨ Email processor optimization complete!")
            return optimized_processor
            
        except Exception as e:
            logger.error(f"⚠️ Optimization failed: {e}")
            logger.info("📝 Falling back to non-optimized processor")
            # Return basic processor if optimization fails
            return EmailProcessor()
    
    def process_email(self, email_text: str) -> Dict[str, Union[str, bool]]:
        """
        Process a single email through classification and reply generation
        
        This is the main method that external code calls to process emails.
        It provides a simple interface that hides the complexity of DSPy operations.
        
        Args:
            email_text: Complete email content (including headers and body)
            
        Returns:
            Dictionary containing:
            - category: Email classification (Quote Request, New Order Received, etc.)
            - draft_reply: Generated professional reply
            - success: Whether processing was successful
            
        Raises:
            ValueError: If engine is not initialized
        """
        
        # Ensure engine is ready
        if not self.is_initialized:
            raise ValueError(
                "EmailEngine not initialized. Call engine.initialize() first."
            )
        
        try:
            logger.info("📧 Processing email through AI pipeline...")
            
            # Call our optimized DSPy processor
            # This will:
            # 1. Classify the email using Gemini AI + our training examples
            # 2. Generate a reply using Chain of Thought reasoning
            result = self.processor(email_text=email_text)
            
            logger.info(f"✅ Email classified as: {result.category}")
            
            return {
                'category': result.category,
                'draft_reply': result.draft_reply,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing email: {e}")
            
            # Return error information instead of crashing
            return {
                'category': 'Error',
                'draft_reply': f'Error processing email: {str(e)}',
                'success': False
            }


# ================================
# Helper Functions for Email Parsing
# ================================
# These functions extract information from raw email text
# They use regex patterns to find common email elements

def get_current_timestamp() -> str:
    """
    Get current timestamp in standardized format
    
    Returns a consistent timestamp format that works across the system.
    This ensures all dates are stored in the same format to avoid parsing issues.
    
    Returns:
        Current timestamp in YYYY-MM-DD HH:MM:SS format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_subject(email_text: str) -> str:
    """
    Extract the subject line from email content
    
    Looks for "Subject: ..." pattern in the email text.
    This works with most standard email formats.
    
    Args:
        email_text: Complete email content
        
    Returns:
        Subject line text or "No Subject" if not found
    """
    
    # Look for "Subject: " followed by anything until end of line
    # re.IGNORECASE makes it work with "subject:", "SUBJECT:", etc.
    match = re.search(r"Subject: (.*)", email_text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()  # Remove extra whitespace
    else:
        return "No Subject"


def extract_sender_info(email_text: str) -> tuple[str, str]:
    """
    Extract sender's name and email address from the 'From' line
    
    Handles different email formats:
    - "John Doe <john@example.com>"
    - "john@example.com"
    - "John Doe" <john@example.com>
    
    Args:
        email_text: Complete email content
        
    Returns:
        Tuple of (sender_name, sender_email)
        Returns ("Unknown Sender", "N/A") if parsing fails
    """
    
    # Look for "From: " line
    match = re.search(r"From: (.*)", email_text, re.IGNORECASE)
    if not match:
        return "Unknown Sender", "N/A"

    from_line = match.group(1).strip()
    
    # Try to find email in angle brackets: Name <email@domain.com>
    email_match = re.search(r'<([^>]+)>', from_line)
    if email_match:
        email = email_match.group(1)
        # Remove the email part to get just the name
        name = from_line.replace(email_match.group(0), '').strip().replace('"', '')
        return name if name else "Unknown Sender", email
    
    # Try to find email directly if no angle brackets
    email_match = re.search(r'([\w\.\-]+@[\w\.\-]+)', from_line)
    if email_match:
        email = email_match.group(0)
        # Remove email to get name (if any)
        name = from_line.replace(email, '').strip().replace('"', '')
        return name if name else "Unknown Sender", email
        
    # If no email found, treat entire from_line as name
    return from_line, "N/A"


def get_category_emoji(category: str) -> str:
    """
    Get emoji representation for email categories
    
    Makes the UI more visual and user-friendly.
    
    Args:
        category: Email category string
        
    Returns:
        Appropriate emoji for the category
    """
    
    emoji_map = {
        "Quote Request": "💰",  # Money emoji for pricing requests
        "New Order Received": "📦",  # Package emoji for orders
        "Delivery Follow-up": "🚚",  # Truck emoji for delivery
        "Other": "📄"  # Document emoji for general emails
    }
    
    return emoji_map.get(category, "📄")  # Default to document emoji


def get_priority_level(category: str) -> str:
    """
    Determine priority level based on email category
    
    Business logic for email prioritization:
    - New Orders: High priority (revenue generating)
    - Delivery Follow-ups: Medium priority (customer satisfaction)
    - Quote Requests: Medium priority (potential revenue)
    - Other: Low priority (general inquiries)
    
    Args:
        category: Email category string
        
    Returns:
        Priority level: "High", "Medium", or "Low"
    """
    
    priority_map = {
        "New Order Received": "High",     # Money in the bank
        "Delivery Follow-up": "Medium",   # Customer service
        "Quote Request": "Medium",        # Potential money
        "Other": "Low"                    # Everything else
    }
    
    return priority_map.get(category, "Low") 