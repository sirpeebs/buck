# Haas Service Assistant

A Streamlit application that helps Haas CNC field service technicians troubleshoot issues by retrieving relevant information from previous service records.

## Features

- Semantic search across service records
- Intelligent part number and description matching
- Dynamic model filtering based on database contents
- GPT-enhanced responses
- Real-time troubleshooting assistance

## Deployment Instructions

### Option 1: Deploy on Streamlit Community Cloud (Recommended)

1. Create a GitHub repository and push your code
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (haas_service_assistant.py)
6. Add your environment variables:
   - OPENAI_API_KEY
   - PINECONE_API_KEY
   - PINECONE_INDEX

### Option 2: Deploy on Heroku

1. Create a `Procfile` with:
   ```
   web: streamlit run haas_service_assistant.py
   ```
2. Create a Heroku app
3. Set environment variables in Heroku dashboard
4. Deploy using Heroku CLI or GitHub integration

### Option 3: Deploy on AWS Elastic Beanstalk

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "haas_service_assistant.py"]
   ```
2. Create an Elastic Beanstalk application
3. Set environment variables in Elastic Beanstalk configuration
4. Deploy using AWS CLI or GitHub Actions

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX`: Your Pinecone index name

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your environment variables
5. Run the application:
   ```bash
   streamlit run haas_service_assistant.py
   ```

## Data Requirements

The Pinecone database should contain service records with the following metadata fields:
- `Model`: The model of the Haas CNC machine (e.g., "VF-3")
- `Serial`: The serial number of the machine
- `Alarm`: Alarm code number (e.g., "108")
- `WorkRequired`: Description of the issue/work required
- `ServicePerformed`: Detailed description of the service performed to fix the issue
- `VerificationTest`: Description of tests performed to verify the fix

## License

[MIT License](LICENSE) 