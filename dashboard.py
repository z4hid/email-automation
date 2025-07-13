import streamlit as st
import pandas as pd
import os
import time
import urllib.parse
from datetime import datetime
from engine import EmailEngine, extract_subject, extract_sender_info, get_category_emoji, get_priority_level, get_current_timestamp

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Email Processing Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CSV_FILE_PATH = "processed_emails.csv"
BACKUP_FILE_PATH = "processed_emails.csv.backup"

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'engine_initialized' not in st.session_state:
    st.session_state.engine_initialized = False
if 'show_bulk_replies' not in st.session_state:
    st.session_state.show_bulk_replies = False

@st.cache_resource
def get_email_engine():
    """Create and initialize the EmailEngine."""
    engine = EmailEngine()
    engine.initialize()
    return engine

# Helper functions for mailto links
def create_mailto_link(email, subject, body):
    """Create a mailto link with proper URL encoding"""
    if not email or email == "N/A":
        return "#"
    
    # URL encode the subject and body
    encoded_subject = urllib.parse.quote(subject)
    encoded_body = urllib.parse.quote(body)
    
    mailto_link = f"mailto:{email}?subject={encoded_subject}&body={encoded_body}"
    return mailto_link

def format_email_with_mailto(email, subject="", body=""):
    """Format email address as clickable mailto link"""
    if not email or email == "N/A":
        return email
    
    if subject and body:
        mailto_link = create_mailto_link(email, subject, body)
        return f'<a href="{mailto_link}" target="_blank">{email}</a>'
    else:
        return f'<a href="mailto:{email}">{email}</a>'

def create_reply_subject(original_subject):
    """Create a reply subject line"""
    if not original_subject or original_subject == "No Subject":
        return "Re: Your Email"
    
    if original_subject.lower().startswith("re:"):
        return original_subject
    else:
        return f"Re: {original_subject}"

# Data handling functions
def standardize_date_format(date_str):
    """Standardize various date formats to YYYY-MM-DD HH:MM:SS"""
    if pd.isna(date_str) or date_str == "":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Try to parse with flexible format
        parsed_date = pd.to_datetime(date_str, format='mixed', errors='coerce')
        if pd.isna(parsed_date):
            # If parsing failed, use current time
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Convert to standard format
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # Fallback to current time if all else fails
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def migrate_old_csv(df):
    """Migrate old CSV format to new format"""
    try:
        if 'id' in df.columns and 'ID' not in df.columns:
            column_mapping = {
                'id': 'ID', 'date': 'Date', 'name': 'Name', 'email': 'Email',
                'subject': 'Subject', 'category': 'Category', 'status': 'Status',
                'draft_reply': 'Draft Reply', 'original_email': 'Original Email'
            }
            df = df.rename(columns=column_mapping)
            
            if 'Priority' not in df.columns:
                df['Priority'] = df['Category'].apply(lambda x: get_priority_level(x) if pd.notna(x) else 'Low')
            if 'Remarks' not in df.columns:
                df['Remarks'] = 'Migrated from old format'
            
            old_columns = ['created_at', 'updated_at']
            df = df.drop(columns=[col for col in old_columns if col in df.columns])
            
            desired_columns = [
                "ID", "Date", "Name", "Email", "Subject", "Category", 
                "Priority", "Status", "Remarks", "Draft Reply", "Original Email"
            ]
            df = df.reindex(columns=desired_columns)
            
            if df['ID'].dtype == 'object':
                df['ID'] = range(1, len(df) + 1)
            
            st.info("📄 Migrated old CSV format to new format")
        
        # Always standardize date format
        if 'Date' in df.columns and not df.empty:
            df['Date'] = df['Date'].apply(standardize_date_format)
            
        return df
    except Exception as e:
        st.error(f"Error migrating CSV: {e}")
        return df

def load_data_with_sync():
    """Load data from CSV with backup sync"""
    try:
        if os.path.exists(CSV_FILE_PATH):
            df = pd.read_csv(CSV_FILE_PATH)
            df = migrate_old_csv(df)  # This now includes date standardization
            df.to_csv(BACKUP_FILE_PATH, index=False)
            return df
        else:
            df = pd.DataFrame(columns=[
                "ID", "Date", "Name", "Email", "Subject", "Category", 
                "Priority", "Status", "Remarks", "Draft Reply", "Original Email"
            ])
            save_data_with_sync(df)
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Creating fresh data structure...")
        return pd.DataFrame(columns=[
            "ID", "Date", "Name", "Email", "Subject", "Category", 
            "Priority", "Status", "Remarks", "Draft Reply", "Original Email"
        ])

def save_data_with_sync(df):
    """Save data to CSV with backup and sync"""
    try:
        df.to_csv(CSV_FILE_PATH, index=False)
        df.to_csv(BACKUP_FILE_PATH, index=False)
        st.session_state.last_refresh = time.time()
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def get_next_id(df):
    """Get next available ID for new entries"""
    if df.empty:
        return 1
    return df['ID'].max() + 1

def process_new_email(email_engine, email_text, df):
    """Process a new email and add to dataframe"""
    try:
        name, email = extract_sender_info(email_text)
        subject = extract_subject(email_text)
        
        result = email_engine.process_email(email_text)
        
        if result['success']:
            new_entry = {
                "ID": get_next_id(df),
                "Date": get_current_timestamp(),
                "Name": name,
                "Email": email,
                "Subject": subject,
                "Category": result['category'],
                "Priority": get_priority_level(result['category']),
                "Status": "Pending",
                "Remarks": f"Auto-classified as {result['category']}",
                "Draft Reply": result['draft_reply'],
                "Original Email": email_text
            }
            
            new_df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
            return new_df, True, f"Email processed successfully! Classified as **{result['category']}**", new_entry
        else:
            return df, False, f"Error processing email: {result['draft_reply']}", None
            
    except Exception as e:
        return df, False, f"Error processing email: {str(e)}", None

def main():
    # Header
    st.title("📧 Email Processing Dashboard")
    st.markdown("**AI-Powered Email Classification & Response Generation**")
    
    # Initialize engine
    email_engine = get_email_engine()
    if not email_engine.is_initialized:
        st.session_state.engine_initialized = False
        st.error("❌ Failed to initialize AI Engine. Please check your API key.")
        st.stop()
    else:
        st.session_state.engine_initialized = True
    
    # Load data
    df = load_data_with_sync()
    
    # Additional safeguard: Check and fix any date format issues
    if not df.empty and 'Date' in df.columns:
        try:
            # Test if dates can be parsed properly
            pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        except Exception as e:
            st.warning(f"⚠️ Found date format issues, fixing them automatically...")
            df['Date'] = df['Date'].apply(standardize_date_format)
            save_data_with_sync(df)
            st.success("✅ Date formats standardized!")
    
    # Sidebar for quick stats and controls
    with st.sidebar:
        st.header("📊 Quick Stats")
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📬 Total", len(df))
                st.metric("⏳ Pending", len(df[df['Status'] == 'Pending']))
            with col2:
                st.metric("✅ Done", len(df[df['Status'] == 'Done']))
                st.metric("🔄 Progress", len(df[df['Status'] == 'In Progress']))
            
            st.divider()
            
            # Category breakdown
            st.subheader("📋 Categories")
            category_counts = df['Category'].value_counts()
            for category, count in category_counts.items():
                emoji = get_category_emoji(category)
                st.write(f"{emoji} {category}: **{count}**")
            
            st.divider()
            
            # Action Quick Stats
            st.subheader("📬 Action Overview")
            pending_actions = len(df[df['Status'].isin(['Pending', 'In Progress'])])
            replies_ready = len(df[(df['Draft Reply'].notna()) & (df['Draft Reply'] != "No reply needed for this category.") & (df['Status'] != 'Done')])
            
            if pending_actions > 0:
                st.warning(f"⚠️ **{pending_actions}** emails need attention")
            if replies_ready > 0:
                st.info(f"📧 **{replies_ready}** replies ready to send")
            
            if pending_actions == 0 and replies_ready == 0:
                st.success("✅ All emails handled!")
            
            # Navigation tip
            if pending_actions > 0 or replies_ready > 0:
                st.info("💡 Visit the **'📬 Email Actions'** tab for comprehensive action management!")
        else:
            st.info("📭 No emails processed yet")
        
        st.divider()
        
        # Controls
        st.header("⚙️ Controls")
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh", value=st.session_state.auto_refresh)
        
        if st.button("🔄 Refresh Now", use_container_width=True, key="sidebar_refresh"):
            st.rerun()
        
        st.divider()
        
        # Data Management
        st.header("💾 Data Export")
        if not df.empty:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📧 Process New Email", "📊 Email Dashboard", "📬 Email Actions", "⚙️ Settings"])
    
    # Tab 1: Process New Email
    with tab1:
        st.header("📧 Process New Email")
        st.markdown("Paste your email content below and let AI classify it and generate a reply.")
        
        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Email input
            email_input = st.text_area(
                "📮 Paste Email Content:",
                height=400,
                placeholder="""From: john.doe@example.com
Subject: Request for Quote - Custom Parts

Dear Sales Team,

I hope this email finds you well. We are interested in getting a quote for custom sensor assemblies for our upcoming project.

Could you please provide pricing for:
- 50 units of Model XYZ-123
- 100 units of Model ABC-456

Please include delivery timeline and payment terms.

Best regards,
John Doe
Engineering Manager""",
                help="Paste the complete email including From, Subject, and body"
            )
            
            # Process button
            if st.button("🚀 Process Email", type="primary", use_container_width=True, key="process_email_btn"):
                if email_input.strip():
                    with st.spinner("🤖 Processing email with AI..."):
                        df, success, message, new_entry = process_new_email(email_engine, email_input, df)
                        
                        if success:
                            save_data_with_sync(df)
                            st.success(message)
                            
                            # Show processed result
                            if new_entry:
                                st.subheader("📋 Processing Results")
                                
                                result_col1, result_col2 = st.columns(2)
                                with result_col1:
                                    st.info(f"**📧 From:** {new_entry['Name']} ({new_entry['Email']})")
                                    st.info(f"**📄 Subject:** {new_entry['Subject']}")
                                    st.info(f"**🏷️ Category:** {get_category_emoji(new_entry['Category'])} {new_entry['Category']}")
                                
                                with result_col2:
                                    st.info(f"**⚡ Priority:** {new_entry['Priority']}")
                                    st.info(f"**📊 Status:** {new_entry['Status']}")
                                    st.info(f"**🕒 Processed:** {new_entry['Date']}")
                                
                                # Show draft reply
                                st.subheader("🤖 Generated Reply")
                                st.text_area(
                                    "Draft Reply:",
                                    value=new_entry['Draft Reply'],
                                    height=200,
                                    disabled=True
                                )
                                
                                # Email action buttons
                                st.subheader("📬 Email Actions")
                                email_col1, email_col2 = st.columns(2)
                                
                                with email_col1:
                                    # Create mailto link with reply
                                    if new_entry['Email'] and new_entry['Email'] != "N/A":
                                        reply_subject = create_reply_subject(new_entry['Subject'])
                                        mailto_link = create_mailto_link(
                                            new_entry['Email'], 
                                            reply_subject, 
                                            new_entry['Draft Reply']
                                        )
                                        st.markdown(f"""
                                        <a href="{mailto_link}" target="_blank">
                                            <button style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                                                         border: none; border-radius: 5px; cursor: pointer; width: 100%;">
                                                📧 Send Reply Email
                                            </button>
                                        </a>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.warning("⚠️ No valid email address found")
                                
                                with email_col2:
                                    # Copy draft to clipboard button
                                    st.write("📋 Copy Draft Reply:")
                                    st.code(new_entry['Draft Reply'], language=None)
                                
                                # Quick actions
                                st.subheader("⚡ Quick Actions")
                                action_col1, action_col2, action_col3 = st.columns(3)
                                
                                with action_col1:
                                    if st.button("✅ Mark as Done", use_container_width=True, key="tab1_mark_done"):
                                        df.loc[df['ID'] == new_entry['ID'], 'Status'] = 'Done'
                                        save_data_with_sync(df)
                                        st.success("Marked as done!")
                                        st.rerun()
                                
                                with action_col2:
                                    if st.button("🔄 Set In Progress", use_container_width=True, key="tab1_set_progress"):
                                        df.loc[df['ID'] == new_entry['ID'], 'Status'] = 'In Progress'
                                        save_data_with_sync(df)
                                        st.success("Set to in progress!")
                                        st.rerun()
                                
                                with action_col3:
                                    if st.button("📬 Manage Actions", use_container_width=True, key="tab1_manage_actions"):
                                        st.info("💡 Tip: Switch to the 'Email Actions' tab to manage all your email replies and actions in one place!")
                        else:
                            st.error(message)
                else:
                    st.warning("⚠️ Please enter email content to process")
        
        with col2:
            st.subheader("📖 How to Use")
            st.markdown("""
            **Steps:**
            1. Copy the complete email content
            2. Paste it in the text area
            3. Click "Process Email"
            4. Review the AI classification and reply
            5. Click "Send Reply Email" to open your email client
            6. Use quick actions or view in dashboard
            
            **Email Format:**
            ```
            From: sender@example.com
            Subject: Email subject
            
            Email body content...
            ```
            """)
            
            # Recent activity
            if not df.empty:
                st.subheader("📈 Recent Activity")
                recent_emails = df.head(5)
                for _, email in recent_emails.iterrows():
                    with st.expander(f"{get_category_emoji(email['Category'])} {email['Subject'][:30]}..."):
                        st.write(f"**From:** {email['Name']}")
                        st.write(f"**Category:** {email['Category']}")
                        st.write(f"**Status:** {email['Status']}")
                        st.write(f"**Date:** {email['Date']}")
    
    # Tab 2: Email Dashboard
    with tab2:
        st.header("📊 Email Dashboard")
        
        if not df.empty:
            # Filters
            st.subheader("🔍 Filters")
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                status_filter = st.selectbox("📊 Status:", ["All"] + df['Status'].unique().tolist())
            
            with filter_col2:
                category_filter = st.selectbox("🏷️ Category:", ["All"] + df['Category'].unique().tolist())
            
            with filter_col3:
                priority_filter = st.selectbox("⚡ Priority:", ["All"] + df['Priority'].unique().tolist())
            
            with filter_col4:
                date_filter = st.date_input("📅 Date Filter:", value=None)
            
            # Apply filters
            filtered_df = df.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['Status'] == status_filter]
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['Category'] == category_filter]
            if priority_filter != "All":
                filtered_df = filtered_df[filtered_df['Priority'] == priority_filter]
            if date_filter:
                try:
                    # Handle multiple date formats flexibly
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['Date'], format='mixed', errors='coerce').dt.date == date_filter]
                except Exception as e:
                    st.warning(f"⚠️ Date filtering error: {e}. Showing all dates.")
                    # Don't filter by date if there's an error
            
            # Display count
            st.info(f"📋 Showing **{len(filtered_df)}** of **{len(df)}** emails")
            
            # Data editor
            st.subheader("✏️ Email Data")
            
            # Configure columns
            display_columns = {
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Date": st.column_config.TextColumn("📅 Date", width="medium"),
                "Name": st.column_config.TextColumn("👤 Name", width="medium"),
                "Email": st.column_config.TextColumn("📧 Email", width="medium"),
                "Subject": st.column_config.TextColumn("📄 Subject", width="large"),
                "Category": st.column_config.TextColumn("🏷️ Category", width="medium"),
                "Priority": st.column_config.SelectboxColumn(
                    "⚡ Priority",
                    options=["High", "Medium", "Low"],
                    width="small"
                ),
                "Status": st.column_config.SelectboxColumn(
                    "📊 Status",
                    options=["Pending", "In Progress", "Done", "On Hold"],
                    width="medium"
                ),
                "Remarks": st.column_config.TextColumn("📝 Remarks", width="large"),
                "Draft Reply": st.column_config.TextColumn("🤖 Draft Reply", width="large"),
                "Original Email": None
            }
            
            # Editable dataframe
            edited_df = st.data_editor(
                filtered_df,
                column_config=display_columns,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="email_editor"
            )
            
            # Auto-save changes
            if not filtered_df.equals(edited_df):
                for idx, row in edited_df.iterrows():
                    mask = df['ID'] == row['ID']
                    for col in edited_df.columns:
                        if col in df.columns:
                            df.loc[mask, col] = row[col]
                
                if save_data_with_sync(df):
                    st.success("✅ Changes saved automatically!")
                    time.sleep(1)
                    st.rerun()
            
            # Quick Actions Reference
            st.subheader("📬 Quick Actions")
            
            # Count actionable emails
            actionable_emails = filtered_df[
                (filtered_df['Email'] != "N/A") & 
                (filtered_df['Draft Reply'].notna()) & 
                (filtered_df['Draft Reply'] != "No reply needed for this category.") &
                (filtered_df['Status'] != 'Done')
            ]
            
            if not actionable_emails.empty:
                st.info(f"🚀 **{len(actionable_emails)}** emails are ready for action!")
                
                action_ref_col1, action_ref_col2 = st.columns(2)
                
                with action_ref_col1:
                    st.markdown("""
                    **📧 Email Actions Available:**
                    - Send replies with pre-filled content
                    - Edit AI-generated responses
                    - Track email status and progress
                    - Bulk action management
                    """)
                
                with action_ref_col2:
                    st.markdown("""
                    **💡 Pro Tip:**
                    Switch to the **'📬 Email Actions'** tab for:
                    - Advanced filtering and search
                    - Comprehensive action management
                    - Bulk operations
                    - Priority-based sorting
                    """)
                
                # Show a sample of pending high-priority emails
                high_priority = actionable_emails[actionable_emails['Priority'] == 'High']
                if not high_priority.empty:
                    st.warning(f"⚠️ **{len(high_priority)}** high-priority emails need immediate attention!")
                    
                    for _, row in high_priority.head(3).iterrows():  # Show top 3 high priority
                        with st.expander(f"🔥 HIGH PRIORITY: {row['Name']} - {row['Subject'][:30]}..."):
                            st.write(f"**Category:** {get_category_emoji(row['Category'])} {row['Category']}")
                            st.write(f"**Date:** {row['Date']}")
                            
                            reply_subject = create_reply_subject(row['Subject'])
                            mailto_link = create_mailto_link(row['Email'], reply_subject, row['Draft Reply'])
                            
                            st.markdown(f"""
                            <a href="{mailto_link}" target="_blank">
                                <button style="background-color: #FF4B4B; color: white; padding: 10px 20px; 
                                             border: none; border-radius: 5px; cursor: pointer; width: 100%;
                                             font-weight: bold;">
                                    🚨 URGENT: Send Reply Now
                                </button>
                            </a>
                            """, unsafe_allow_html=True)
            else:
                st.success("✅ No pending email actions for the current filter!")
                st.balloons()
            
            # Bulk actions
            st.subheader("🔧 Bulk Actions")
            bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
            
            with bulk_col1:
                if st.button("✅ Mark All as Done", use_container_width=True, key="tab2_mark_all_done"):
                    df.loc[df['Status'] != 'Done', 'Status'] = 'Done'
                    save_data_with_sync(df)
                    st.success("All emails marked as done!")
                    st.rerun()
            
            with bulk_col2:
                if st.button("🔄 Reset to Pending", use_container_width=True, key="tab2_reset_pending"):
                    df['Status'] = 'Pending'
                    save_data_with_sync(df)
                    st.success("All emails reset to pending!")
                    st.rerun()
            
            with bulk_col3:
                if st.button("🗑️ Delete Completed", use_container_width=True, key="tab2_delete_completed"):
                    df = df[df['Status'] != 'Done']
                    save_data_with_sync(df)
                    st.success("Completed emails deleted!")
                    st.rerun()
        
        else:
            st.info("📭 No emails in the system yet. Go to the 'Process New Email' tab to get started!")
    
    # Tab 3: Email Actions
    with tab3:
        st.header("📬 Email Actions Center")
        st.markdown("**Manage all your email replies and actions in one place**")
        
        if not df.empty:
            # Action filters and search
            st.subheader("🔍 Action Filters & Search")
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                action_status_filter = st.selectbox("📊 Action Status:", 
                    ["All", "Pending Actions", "Completed Actions", "In Progress"])
            
            with action_col2:
                action_category_filter = st.selectbox("🏷️ Email Category:", 
                    ["All"] + df['Category'].unique().tolist())
            
            with action_col3:
                action_priority_filter = st.selectbox("⚡ Priority Level:", 
                    ["All", "High", "Medium", "Low"])
            
            with action_col4:
                search_term = st.text_input("🔎 Search emails:", 
                    placeholder="Search by name, email, subject...")
            
            # Apply filters for actions
            action_df = df.copy()
            
            # Filter by action status
            if action_status_filter == "Pending Actions":
                action_df = action_df[action_df['Status'].isin(['Pending', 'In Progress'])]
            elif action_status_filter == "Completed Actions":
                action_df = action_df[action_df['Status'] == 'Done']
            elif action_status_filter == "In Progress":
                action_df = action_df[action_df['Status'] == 'In Progress']
            
            # Filter by category
            if action_category_filter != "All":
                action_df = action_df[action_df['Category'] == action_category_filter]
            
            # Filter by priority
            if action_priority_filter != "All":
                action_df = action_df[action_df['Priority'] == action_priority_filter]
            
            # Search functionality
            if search_term:
                search_mask = (
                    action_df['Name'].str.contains(search_term, case=False, na=False) |
                    action_df['Email'].str.contains(search_term, case=False, na=False) |
                    action_df['Subject'].str.contains(search_term, case=False, na=False) |
                    action_df['Remarks'].str.contains(search_term, case=False, na=False)
                )
                action_df = action_df[search_mask]
            
            # Display results count
            st.info(f"📋 Found **{len(action_df)}** emails matching your criteria")
            
            if not action_df.empty:
                # Bulk Actions
                st.subheader("🔧 Bulk Actions")
                bulk_action_col1, bulk_action_col2, bulk_action_col3, bulk_action_col4 = st.columns(4)
                
                with bulk_action_col1:
                    if st.button("📧 Generate All Reply Links", use_container_width=True, key="tab3_generate_links"):
                        st.session_state.show_bulk_replies = True
                
                with bulk_action_col2:
                    if st.button("✅ Mark All as Done", use_container_width=True, key="tab3_mark_all_done"):
                        for idx in action_df.index:
                            df.loc[idx, 'Status'] = 'Done'
                        save_data_with_sync(df)
                        st.success("All selected emails marked as done!")
                        st.rerun()
                
                with bulk_action_col3:
                    if st.button("🔄 Set All In Progress", use_container_width=True, key="tab3_set_all_progress"):
                        for idx in action_df.index:
                            df.loc[idx, 'Status'] = 'In Progress'
                        save_data_with_sync(df)
                        st.success("All selected emails set to in progress!")
                        st.rerun()
                
                with bulk_action_col4:
                    if st.button("⏳ Reset to Pending", use_container_width=True, key="tab3_reset_pending"):
                        for idx in action_df.index:
                            df.loc[idx, 'Status'] = 'Pending'
                        save_data_with_sync(df)
                        st.success("All selected emails reset to pending!")
                        st.rerun()
                
                st.divider()
                
                # Individual Actions
                st.subheader("📋 Individual Email Actions")
                
                # Sort by priority and date
                priority_order = {"High": 0, "Medium": 1, "Low": 2}
                action_df['Priority_Sort'] = action_df['Priority'].map(priority_order)
                action_df = action_df.sort_values(['Priority_Sort', 'Date'], ascending=[True, False])
                action_df = action_df.drop('Priority_Sort', axis=1)
                
                for idx, row in action_df.iterrows():
                    with st.expander(f"{get_category_emoji(row['Category'])} {row['Priority']} Priority - {row['Name']} | {row['Subject'][:50]}..."):
                        # Email details
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.write(f"**👤 From:** {row['Name']}")
                            st.write(f"**📧 Email:** {row['Email']}")
                            st.write(f"**📄 Subject:** {row['Subject']}")
                            st.write(f"**🏷️ Category:** {get_category_emoji(row['Category'])} {row['Category']}")
                        
                        with detail_col2:
                            st.write(f"**⚡ Priority:** {row['Priority']}")
                            st.write(f"**📊 Status:** {row['Status']}")
                            st.write(f"**🕒 Date:** {row['Date']}")
                            st.write(f"**📝 Remarks:** {row['Remarks']}")
                        
                        # Draft reply preview
                        if row['Draft Reply'] and row['Draft Reply'] != "No reply needed for this category.":
                            st.subheader("🤖 AI Generated Reply")
                            reply_preview = row['Draft Reply'][:200] + "..." if len(row['Draft Reply']) > 200 else row['Draft Reply']
                            st.text_area("Preview:", value=reply_preview, height=100, disabled=True, key=f"preview_{idx}")
                            
                            # Action buttons
                            action_button_col1, action_button_col2, action_button_col3, action_button_col4 = st.columns(4)
                            
                            with action_button_col1:
                                if row['Email'] and row['Email'] != "N/A":
                                    reply_subject = create_reply_subject(row['Subject'])
                                    mailto_link = create_mailto_link(row['Email'], reply_subject, row['Draft Reply'])
                                    st.markdown(f"""
                                    <a href="{mailto_link}" target="_blank">
                                        <button style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                                                     border: none; border-radius: 4px; cursor: pointer; width: 100%;
                                                     margin-bottom: 5px;">
                                            📧 Send Reply
                                        </button>
                                    </a>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ No valid email")
                            
                            with action_button_col2:
                                if st.button("✅ Mark Done", key=f"done_{idx}", use_container_width=True):
                                    df.loc[idx, 'Status'] = 'Done'
                                    df.loc[idx, 'Remarks'] = f"Completed on {get_current_timestamp()}"
                                    save_data_with_sync(df)
                                    st.success("Marked as done!")
                                    st.rerun()
                            
                            with action_button_col3:
                                if st.button("🔄 In Progress", key=f"progress_{idx}", use_container_width=True):
                                    df.loc[idx, 'Status'] = 'In Progress'
                                    df.loc[idx, 'Remarks'] = f"Started working on {get_current_timestamp()}"
                                    save_data_with_sync(df)
                                    st.success("Set to in progress!")
                                    st.rerun()
                            
                            with action_button_col4:
                                if st.button("📝 Edit Reply", key=f"edit_{idx}", use_container_width=True):
                                    st.session_state[f"edit_mode_{idx}"] = True
                                    st.rerun()
                            
                            # Edit mode
                            if st.session_state.get(f"edit_mode_{idx}", False):
                                st.subheader("✏️ Edit Draft Reply")
                                new_reply = st.text_area("Edit your reply:", value=row['Draft Reply'], height=150, key=f"edit_reply_{idx}")
                                
                                edit_col1, edit_col2 = st.columns(2)
                                with edit_col1:
                                    if st.button("💾 Save Changes", key=f"save_{idx}"):
                                        df.loc[idx, 'Draft Reply'] = new_reply
                                        df.loc[idx, 'Remarks'] = f"Reply edited on {get_current_timestamp()}"
                                        save_data_with_sync(df)
                                        st.session_state[f"edit_mode_{idx}"] = False
                                        st.success("Reply updated!")
                                        st.rerun()
                                
                                with edit_col2:
                                    if st.button("❌ Cancel", key=f"cancel_{idx}"):
                                        st.session_state[f"edit_mode_{idx}"] = False
                                        st.rerun()
                        
                        else:
                            st.info("ℹ️ No reply generated for this email category")
                            
                            # Status change buttons for non-reply emails
                            status_col1, status_col2 = st.columns(2)
                            with status_col1:
                                if st.button("✅ Mark Done", key=f"done_no_reply_{idx}", use_container_width=True):
                                    df.loc[idx, 'Status'] = 'Done'
                                    df.loc[idx, 'Remarks'] = f"Completed on {get_current_timestamp()}"
                                    save_data_with_sync(df)
                                    st.success("Marked as done!")
                                    st.rerun()
                            
                            with status_col2:
                                if st.button("🔄 In Progress", key=f"progress_no_reply_{idx}", use_container_width=True):
                                    df.loc[idx, 'Status'] = 'In Progress'
                                    df.loc[idx, 'Remarks'] = f"Started working on {get_current_timestamp()}"
                                    save_data_with_sync(df)
                                    st.success("Set to in progress!")
                                    st.rerun()
                
                # Bulk reply links section
                if st.session_state.get('show_bulk_replies', False):
                    st.subheader("📧 Bulk Reply Links")
                    st.markdown("**Click any link below to open your email client with the pre-filled reply:**")
                    
                    for idx, row in action_df.iterrows():
                        if row['Email'] and row['Email'] != "N/A" and row['Draft Reply'] and row['Draft Reply'] != "No reply needed for this category.":
                            reply_subject = create_reply_subject(row['Subject'])
                            mailto_link = create_mailto_link(row['Email'], reply_subject, row['Draft Reply'])
                            
                            st.markdown(f"""
                            **{get_category_emoji(row['Category'])} {row['Name']}** - {row['Subject'][:40]}...  
                            📧 [Send Reply to {row['Email']}]({mailto_link})
                            """)
                    
                    if st.button("❌ Hide Bulk Links", use_container_width=True, key="tab3_hide_bulk_links"):
                        st.session_state.show_bulk_replies = False
                        st.rerun()
            
            else:
                st.info("🔍 No emails match your current filters. Try adjusting your search criteria.")
                
            # Quick Action Statistics
            st.divider()
            st.subheader("📊 Action Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                pending_count = len(df[df['Status'] == 'Pending'])
                st.metric("⏳ Pending Actions", pending_count)
            
            with stats_col2:
                in_progress_count = len(df[df['Status'] == 'In Progress'])
                st.metric("🔄 In Progress", in_progress_count)
            
            with stats_col3:
                done_count = len(df[df['Status'] == 'Done'])
                st.metric("✅ Completed", done_count)
            
            with stats_col4:
                reply_count = len(df[(df['Draft Reply'].notna()) & (df['Draft Reply'] != "No reply needed for this category.")])
                st.metric("📧 Replies Available", reply_count)
        
        else:
            st.info("📭 No emails in the system yet. Process some emails first to see actions here!")
    
    # Tab 4: Settings
    with tab4:
        st.header("⚙️ Settings & Data Management")
        
        # System status
        st.subheader("🔧 System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            engine_status = "✅ Ready" if st.session_state.engine_initialized else "❌ Not Ready"
            st.metric("AI Engine", engine_status)
        
        with status_col2:
            csv_status = "✅ Found" if os.path.exists(CSV_FILE_PATH) else "❌ Missing"
            st.metric("CSV File", csv_status)
        
        with status_col3:
            backup_status = "✅ Available" if os.path.exists(BACKUP_FILE_PATH) else "❌ Missing"
            st.metric("Backup", backup_status)
        
        st.divider()
        
        # Data management
        st.subheader("💾 Data Management")
        
        mgmt_col1, mgmt_col2 = st.columns(2)
        
        with mgmt_col1:
            st.markdown("**Export Data**")
            if not df.empty:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv_data,
                    file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No data to export")
        
        with mgmt_col2:
            st.markdown("**Restore Data**")
            if st.button("🔄 Restore from Backup", use_container_width=True, key="tab4_restore_backup"):
                if os.path.exists(BACKUP_FILE_PATH):
                    try:
                        backup_df = pd.read_csv(BACKUP_FILE_PATH)
                        save_data_with_sync(backup_df)
                        st.success("✅ Restored from backup!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error restoring backup: {e}")
                else:
                    st.error("No backup file found")
        
        st.divider()
        
        # Danger zone
        st.subheader("⚠️ Danger Zone")
        with st.expander("🚨 Clear All Data"):
            st.warning("This action cannot be undone!")
            confirm_text = st.text_input("Type 'DELETE ALL' to confirm:")
            if st.button("🗑️ Delete All Data", type="secondary", key="tab4_delete_all"):
                if confirm_text == "DELETE ALL":
                    empty_df = pd.DataFrame(columns=[
                        "ID", "Date", "Name", "Email", "Subject", "Category", 
                        "Priority", "Status", "Remarks", "Draft Reply", "Original Email"
                    ])
                    save_data_with_sync(empty_df)
                    st.success("All data cleared!")
                    st.rerun()
                else:
                    st.error("Please type 'DELETE ALL' to confirm")
    
    # Footer
    st.divider()
    st.caption(f"📧 Email Processing Dashboard | Last updated: {get_current_timestamp()}")
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(1)
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 30:
            st.rerun()

if __name__ == "__main__":
    main() 