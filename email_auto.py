import dspy
from dspy.teleprompt import LabeledFewShot

import os

# --- 1. Setup Language Model ---
# Configure the language model you want to use.
# It's good practice to do this at the beginning.
lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
dspy.configure(lm=lm)


# --- 2. Define Signatures ---
# Keep signatures clear and separate. They define the core tasks for your AI.

class EmailClassifierSignature(dspy.Signature):
    """Classify the email into: Quote Request, New Order Received, Delivery Follow-up, or Other."""
    email_text = dspy.InputField(desc="The full content of the email.")
    category = dspy.OutputField(desc="The determined category of the email e.g.  quote, PO, purchase order, new order, RFQ, ETD, delivery date or other.")

class EmailReplySignature(dspy.Signature):
    """Write a professional and helpful email reply based on its category and content."""
    email_category = dspy.InputField(desc="The category of the email.")
    email_content = dspy.InputField(desc="The full content of the original email.")
    draft_reply = dspy.OutputField(desc="The generated draft email reply.")


# --- 3. Create a Cohesive Email Processing Module ---
# This module encapsulates the entire workflow: classify and then reply.
# This is a much cleaner and more reusable structure.

class EmailProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Encapsulate the classifier and reply generator within the main module.
        self.classifier = dspy.Predict(EmailClassifierSignature)
        self.reply_generator = dspy.ChainOfThought(EmailReplySignature)

    def forward(self, email_text):
        # The forward method defines the logic flow.
        classification = self.classifier(email_text=email_text)
        
        # Conditionally generate a reply.
        if classification.category != "Other":
            reply = self.reply_generator(email_category=classification.category, email_content=email_text)
            return dspy.Prediction(category=classification.category, draft_reply=reply.draft_reply)
        
        # Return the category even if no reply is generated.
        return dspy.Prediction(category=classification.category, draft_reply="No reply needed.")


# --- 4. Prepare Training Data and Optimize ---
# Your training examples are essential for guiding the model.
train_examples = [
    dspy.Example(email_text="Subject: Re: Adaptor Cable Quote\n\nIf there is anyway to expedite this one please let me know...", category="Delivery Follow-up").with_inputs("email_text"),
    dspy.Example(email_text="Subject: REM Australia RFQ 10_07_2025\n\nCould you please reconfirm pricing for following...", category="Quote Request").with_inputs("email_text"),
    dspy.Example(email_text="Subject: Purchase Order PO12345\n\nPlease see attached our new purchase order.", category="New Order Received").with_inputs("email_text")
]

# Instantiate your main module.
email_processor = EmailProcessor()

# Compile the module with a teleprompter for optimization.

teleprompter = LabeledFewShot(k=3)
optimized_processor = teleprompter.compile(email_processor, trainset=train_examples)


# --- 5. Execute the Processor ---
# Now, you can process new emails with a single call.
new_email_content = """
From: Brian O'Connell brian.oconnell@apexindustrial.com.au
Sent: Wednesday, July 16, 2025 2:40 PM
To: Jennifer Hale j.hale@synergycomponents.net
Cc: Apex Purchasing purchasing@apexindustrial.com.au
Subject: Purchase Order PO-2025-781 for Quote Q-9942

Hi Jennifer,

Thank you for the prompt quotation.

Please find attached our official Purchase Order PO-2025-781 for 150 units of SC-4500-B, as per your quote Q-9942.

Kindly send an order acknowledgement at your earliest convenience.

Kind regards,

Brian O'Connell
Senior Purchasing Officer

Apex Industrial Solutions
Unit 5, 123 Industrial Drive, Welshpool WA 6106
T +61 8 9331 4000
E brian.oconnell@apexindustrial.com.au | W www.apexindustrial.com.au
"""

# Get the prediction from the optimized processor.
result = optimized_processor(email_text=new_email_content)

print(f"📧 Email classified as: **{result.category}**")
print("\n--- ✍️ Generated Draft Reply ---")
print(result.draft_reply)