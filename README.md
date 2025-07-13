# 📧 AI Email Helper Dashboard

This is a simple web app that uses AI to help you manage your emails. You paste an email into it, and the AI will figure out what the email is about (like a quote request or a new order) and write a draft reply for you.

Everything is shown on a clean dashboard where you can track your work. The app saves your data to a CSV file so you don't lose it.

-----

## ✨ What It Does

  * **Reads Emails for You**: Figures out if an email is a quote request, a new order, or a delivery question.
  * **Writes Replies**: Automatically creates a draft reply that you can use.
  * **Easy Dashboard**: Shows all your processed emails in one place.
  * **Lets You Edit**: You can change the status or add notes for any email right on the dashboard.
  * **Saves Your Work**: Keeps all your data safe in a `processed_emails.csv` file.

-----

## 🚀 How to Set It Up

Follow these steps to get the app running on your computer.

### Step 1: Get the Code

First, you need to copy the project files to your computer. You can do this by downloading the ZIP file or using Git.

### Step 2: Install `uv` (A Python Tool)

We use a tool called `uv` to handle the code packages for this project. It's fast and easy. If you don't have it, open your terminal and run one of these commands:

  * **On Mac or Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  * **On Windows:**
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Step 3: Set Up the Project

1.  **Open a terminal** in the project folder.

2.  **Create a virtual environment:** This makes a separate space for our project's code packages.

    ```bash
    uv venv
    ```

3.  **Activate the environment:**

      * **On Mac or Linux:** `source .venv/bin/activate`
      * **On Windows:** `.venv\Scripts\activate`

4.  **Install the needed packages:**

    ```bash
    uv pip install dspy-ai streamlit python-dotenv pandas
    ```

### Step 4: Add Your API Key

The AI needs an API key to work.

1.  Create a new file in the project folder and name it **`.env`**.
2.  Open the `.env` file and add your key like this:
    ```
    GEMINI_API_KEY="paste_your_key_here"
    ```

-----

## ▶️ How to Run the App

Make sure your virtual environment is still active, then run this command:

```bash
streamlit run app.py
```

Your web browser should open with the app running. That's it\!

-----

## 📁 Files Explained

  * `.venv/`: The virtual space for our project.
  * `.env`: Where you store your secret API key.
  * `processed_emails.csv`: The file where your dashboard data is saved.
  * `app.py`: The main code that runs the app.
  * `README.md`: The file you are reading right now.