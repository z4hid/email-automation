import streamlit as st
import dspy
import os
import pandas as pd
from datetime import datetime
import re
from dotenv import load_dotenv

# --- App Configuration & Secrets ---
# Load environment variables from a .env file for local development
load_dotenv()

# Set the page configuration. This must be the first Streamlit command.
st.set_page_config(layout="wide", page_title="Email Processing Dashboard")

# Define the path for the CSV database file
CSV_FILE_PATH = "processed_emails.csv"

# It's best practice to manage API keys using Streamlit's secrets management for deployment.
# For local development, we are using a .env file.
try:
    # Fallback for local development using .env file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

except (FileNotFoundError, KeyError):
    # Primary method for deployed Streamlit apps
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# --- DSPy Configuration ---
# This function sets up the DSPy language model.
# Using st.cache_resource ensures this expensive setup runs only once.
@st.cache_resource
def setup_dspy_lm():
    """Initializes and configures the DSPy Language Model."""
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Please create a .env file with GEMINI_API_KEY='your_key' or set it in Streamlit secrets.")
        return None
    # Use a more robust model for better instruction following in a complex app.
    lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY)
    dspy.configure(lm=lm)
    return lm

# --- DSPy Signatures and Module ---
# These are the core definitions for our AI's tasks.

class EmailClassifierSignature(dspy.Signature):
    """Classify the email into: Quote Request, New Order Received, Delivery Follow-up, or Other."""
    email_text = dspy.InputField(desc="The full content of the email.")
    category = dspy.OutputField(desc="The most likely category for the email.")

class EmailReplySignature(dspy.Signature):
    """Write a professional and helpful email reply based on its category and content."""
    email_category = dspy.InputField(desc="The category of the email.")
    email_content = dspy.InputField(desc="The full content of the original email.")
    draft_reply = dspy.OutputField(desc="The generated draft email reply.")

class EmailProcessor(dspy.Module):
    """A DSPy module that classifies an email and then generates a reply."""
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(EmailClassifierSignature)
        self.reply_generator = dspy.ChainOfThought(EmailReplySignature)

    def forward(self, email_text):
        classification = self.classifier(email_text=email_text)
        
        if classification.category != "Other":
            reply = self.reply_generator(email_category=classification.category, email_content=email_text)
            return dspy.Prediction(category=classification.category, draft_reply=reply.draft_reply)
        
        return dspy.Prediction(category=classification.category, draft_reply="No reply needed for this category.")

# --- DSPy Optimization ---
# This function compiles the DSPy module.
@st.cache_resource
def compile_processor():
    """Compiles the EmailProcessor module with few-shot examples."""
    train_examples = [
    # Based on Email 1
        dspy.Example(
            email_text="Subject: RFQ - Costing for Custom Sensor Assemblies\n\nHello Sales,\n\nWe require a formal RFQ for a new project. Could you please provide costing for the custom sensor assemblies outlined in the attached drawings (QD-DWG-77A and QD-DWG-77B)?\n\nPlease price for a batch of 50 and a batch of 100.",
            category="Quote Request"
        ).with_inputs("email_text"),

        # Based on Email 2
        dspy.Example(
            email_text="Subject: Request for Quote - Replacement Motor Looms\n\nHi Team,\n\nCould you please provide a quote for the following replacement parts for our weaving machines?\n\nITEM: AI-ML-V4, Industrial Motor Looms\nQUANTITY: 10 units",
            category="Quote Request"
        ).with_inputs("email_text"),

        # Based on Email 3
        dspy.Example(
            email_text="Subject: Purchase Order PO2025-095 for Sensor Assemblies\n\nHi Jennifer,\n\nThank you for the quick turnaround on the quote.\n\nPlease see the attached Purchase Order PO2025-095 for the batch of 100 units. This order attached is based on your quotation Q-9981.",
            category="New Order Received"
        ).with_inputs("email_text"),

        # Based on Email 4
        dspy.Example(
            email_text="Subject: Delivery Inquiry for PO-PW-1134\n\nHello Brian,\n\nI'm following up on our order for motor looms, PO-PW-1134. Can you please provide an estimated delivery date? We need to schedule technicians for the installation and need to know the expected arrival at our facility.",
            category="Delivery Follow-up"
        ).with_inputs("email_text"),

        # Based on Email 5
        dspy.Example(
            email_text="Subject: Quote Refresh - Part SC-3100-D Power Converters\n\nHi Jennifer,\n\nCould you please provide a refreshed quote for 200 units of part SC-3100-D Power Converters? Our last PO was in January, and we just need to verify the current costing before issuing a new order.",
            category="Quote Request"
        ).with_inputs("email_text"),

        # Based on Email 6
        dspy.Example(
            email_text="Subject: New PO Attached - PO-2025-790 for SC-3100-D\n\nHi Jennifer,\n\nThanks for sending that over.\n\nPlease find our new PO attached for the power converters. The order attached, PO-2025-790, is for 200 units.",
            category="New Order Received"
        ).with_inputs("email_text"),

        # Based on Email 7
        dspy.Example(
            email_text="Subject: ETD Request for PO2025-095\n\nHi Jennifer,\n\nCould you provide the target ETD for our order PO2025-095? Our logistics team is planning the receiving schedule and needs to know the estimated date of dispatch from your facility.",
            category="Delivery Follow-up"
        ).with_inputs("email_text"),

        # Based on Email 8
        dspy.Example(
            email_text="Subject: RFQ for Sub-Component Machining - Ref: Our Assemblies SC-8900\n\nHi Aisha,\n\nHope you're having a good week.\n\nWe are sending an RFQ for a new project. Per our conversation, please see attached drawings for a machined sub-component. We need pricing for these parts which will be used in our SC-8900 assemblies.",
            category="Quote Request"  # This is a B2B RFQ, which falls under Quote Request.
        ).with_inputs("email_text"),

        # Based on Email 9
        dspy.Example(
            email_text="Subject: Confirmation for order PO-2025-790\n\nHi Jennifer,\n\nJust a quick email to confirm that order PO-2025-790 is on track to ship out this Friday as per the acknowledged delivery date.\n\nPlease let me know if there are any delays.",
            category="Delivery Follow-up"
        ).with_inputs("email_text"),

        # Based on Email 10
        dspy.Example(
            email_text="Subject: Purchase Order for additional Motor Looms - PO-PW-1145\n\nHello,\n\nPlease accept the attached Purchase Order for an additional 5 units of the AI-ML-V4 Motor Looms. This is a follow-up to our last PO (PO-PW-1134).",
            category="New Order Received"
        ).with_inputs("email_text")
    ]

    
    email_processor = EmailProcessor()
    teleprompter = dspy.teleprompt.LabeledFewShot(k=3)
    optimized_processor = teleprompter.compile(email_processor, trainset=train_examples)
    return optimized_processor

# --- Data Handling Functions ---
def load_data(file_path):
    """Loads data from a CSV file, creating it if it doesn't exist."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Create an empty DataFrame with the correct structure if the file is not found
        return pd.DataFrame(columns=["Date", "Name", "Email", "Subject", "Status", "Remarks", "Draft Reply", "Original Email"])

def save_data(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# --- Helper Functions ---
def extract_subject(email_text):
    """Extracts the subject line from the email content."""
    match = re.search(r"Subject: (.*)", email_text, re.IGNORECASE)
    return match.group(1).strip() if match else "No Subject"

def extract_sender_info(email_text):
    """Extracts sender's name and email address from the 'From' line."""
    match = re.search(r"From: (.*)", email_text, re.IGNORECASE)
    if not match:
        return "Unknown Sender", "N/A"

    from_line = match.group(1).strip()
    
    # Try to find name and email in "Name <email@domain.com>" format
    email_match = re.search(r'<([^>]+)>', from_line)
    if email_match:
        email = email_match.group(1)
        name = from_line.replace(email_match.group(0), '').strip().replace('"', '')
        return name if name else "Unknown Sender", email
    
    # Try to find email directly if no angle brackets
    email_match = re.search(r'([\w\.\-]+@[\w\.\-]+)', from_line)
    if email_match:
        email = email_match.group(0)
        name = from_line.replace(email, '').strip().replace('"', '')
        return name if name else "Unknown Sender", email
        
    return from_line, "N/A" # If only a name is found

# --- Main Application UI ---
st.title("📧 AI-Powered Email Processing Dashboard")
st.markdown("Paste an email below to classify it and generate a draft reply using DSPy.")

# Initialize DSPy LM
lm = setup_dspy_lm()

# Load existing data from CSV
df = load_data(CSV_FILE_PATH)

# Check if the LM was set up correctly
if lm:
    # Load the compiled DSPy processor
    optimized_processor = compile_processor()

    # --- Input and Processing Section ---
    col1, col2 = st.columns([2, 1])

    with col1:
        email_input = st.text_area("Paste Full Email Content Here:", height=250, placeholder="From: ...\nSubject: ...\n\nHi team,\n...")
        
        if st.button("Process Email", type="primary", use_container_width=True):
            if email_input.strip():
                with st.spinner("🤖 AI is processing the email..."):
                    try:
                        # Run the DSPy processor
                        result = optimized_processor(email_text=email_input)
                        
                        # Extract sender info
                        name, email = extract_sender_info(email_input)
                        
                        # Prepare data for the dashboard
                        new_email_entry = {
                            "Date": datetime.now().strftime("%b %d, %Y, %I:%M %p"),
                            "Name": name,
                            "Email": email,
                            "Subject": extract_subject(email_input),
                            "Status": "Pending",
                            "Remarks": result.category,
                            "Draft Reply": result.draft_reply,
                            "Original Email": email_input
                        }
                        
                        # Add new entry to the DataFrame
                        new_entry_df = pd.DataFrame([new_email_entry])
                        df = pd.concat([new_entry_df, df], ignore_index=True)
                        
                        # Save the updated data back to the CSV
                        save_data(df, CSV_FILE_PATH)
                        
                        st.success(f"Email processed! Classified as **{result.category}**.")
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.warning("Please paste an email into the text area.")

    with col2:
        st.subheader("Instructions")
        st.info(
            """
            **Setup for Local Use:**
            1.  Install `python-dotenv`: `pip install python-dotenv`
            2.  Create a file named `.env` in the same folder as this script.
            3.  Add your key: `GEMINI_API_KEY='your_api_key'`
            
            **How to Use:**
            1.  **Process Email**: The app saves data to `processed_emails.csv`.
            2.  **Dashboard**: The table below shows all processed emails.
            3.  **Edit**: Click any cell to edit its content. Changes are saved automatically.
            """
        )

    st.divider()

    # --- Dashboard Display Section ---
    st.subheader("Processed Emails Dashboard")

    if not df.empty:
        # Use st.data_editor to make the DataFrame interactive
        edited_df = st.data_editor(
            df,
            column_config={
                "Date": st.column_config.TextColumn("🗓️ Date", width="small"),
                "Name": st.column_config.TextColumn("👤 Name"),
                "Email": st.column_config.TextColumn("✉️ Email"),
                "Subject": st.column_config.TextColumn("📄 Subject", width="medium"),
                "Status": st.column_config.SelectboxColumn(
                    "📊 Status",
                    options=["Pending", "In Progress", "Done", "On Hold"],
                    required=True,
                ),
                "Remarks": st.column_config.TextColumn("📌 Remarks"),
                "Draft Reply": st.column_config.TextColumn("🤖 Draft Reply", width="large"),
                "Original Email": None # Hide the original email column from the main view
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        
        # If the dataframe has been changed by the user, save it
        if not df.equals(edited_df):
            save_data(edited_df, CSV_FILE_PATH)
            st.rerun() 
    else:
        st.write("No emails processed yet. Paste an email above to get started.")

else:
    st.warning("Could not initialize the Language Model. Please check your API key configuration.")
