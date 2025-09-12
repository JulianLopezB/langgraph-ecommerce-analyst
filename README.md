# LangGraph Data Analysis Agent

An AI-powered data analysis agent that processes e-commerce data from Google BigQuery and generates actionable business insights through natural language interactions.

## 🚀 Features

- **Natural Language Queries**: Ask business questions in plain English
- **Dynamic SQL Generation**: Automatically generates optimized BigQuery queries
- **Advanced Analytics**: Performs customer segmentation, sales forecasting, statistical analysis
- **Secure Code Execution**: Safely executes generated Python code with sandboxing
- **Interactive CLI**: Conversational interface with session management
- **Comprehensive Analytics**: Statistical analysis, ML models, forecasting, business intelligence

## 📋 Requirements

- Python 3.8+
- Google Cloud account with BigQuery access (1TB free tier)
- Google AI Studio API key for Gemini models

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JulianLopezB/langgraph-ecommerce-analyst.git
   cd langgraph-ecommerce-analyst
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Google Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Google Cloud Configuration (optional)
   GOOGLE_CLOUD_PROJECT=your_gcp_project_id
   
   # BigQuery Configuration (optional)
   BQ_DATASET_ID=bigquery-public-data.thelook_ecommerce

   # Deployment Environment (optional: development|production)
   APP_ENV=development
   
   # LangSmith Tracing (optional - for debugging)
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   ```

4. **Set up Google Cloud authentication** (if using your own project):
   ```bash
   gcloud auth application-default login
   ```

## 🚀 Quick Start

1. **Start the interactive CLI**:
   ```bash
   python main.py
   ```

2. **Ask questions about your data**:
    ```
    🔍 Your question: Segment our customers using RFM analysis
    🔍 Your question: What are the sales trends for the last quarter?
    🔍 Your question: Forecast sales for the next 3 months
    ```

## 🐳 Docker Compose

Run the application alongside its dependencies using Docker Compose.

1. Ensure a `.env` file exists in the project root with the required configuration values.
2. Build and start all services:
   ```bash
   docker-compose up --build
   ```
   This launches the application, a PostgreSQL database, and a RabbitMQ broker on an isolated
   `app-network` and persists data in the `db_data` and `rabbitmq_data` volumes.
3. Stop the services when finished:
   ```bash
   docker-compose down
   ```

## 📊 Example Queries

### Customer Analysis
- "Segment our customers using RFM analysis and identify the most valuable segment"
- "Analyze customer churn patterns and retention rates"
- "What is the customer lifetime value by segment?"

### Sales Analysis
- "Show me sales trends for the last 6 months"
- "Forecast revenue for the next quarter"
- "Which products have the highest revenue growth?"

### Product Analysis
- "What are our top-performing products by revenue?"
- "Analyze product category performance"
- "What products should we recommend to customer ID 12345?"

### Advanced Analytics
- "Perform cohort analysis on customer behavior"
- "Detect anomalies in our sales data"
- "Build a customer churn prediction model"

## 🏗️ Architecture

The system follows a modular, layered architecture:

### Core Components

- **LangGraph Agent**: Orchestrates the analysis workflow
- **Query Understanding**: Classifies user intent and extracts requirements
- **SQL Generator**: Creates optimized BigQuery queries
- **Code Generator**: Generates Python analysis code dynamically
- **Secure Executor**: Safely executes code with resource limits
- **Session Manager**: Handles conversation history and context
- **AnalysisController**: Application layer facade reused by interfaces

### Data Flow

1. **User Input** → Query Understanding
2. **Intent Classification** → SQL Generation
3. **Query Execution** → Data Retrieval
4. **Python Code Generation** → Validation
5. **Secure Execution** → Result Synthesis
6. **Insights Generation** → User Output

## 🔧 Configuration

### Execution Limits
```python
# In config.py
execution_limits:
  max_execution_time: 300  # seconds
  max_memory_mb: 1024     # MB
  max_output_size_mb: 100 # MB
```

### Security Settings
```python
security_settings:
  enable_code_scanning: true
  allowed_imports: [pandas, numpy, matplotlib, sklearn, ...]
  forbidden_patterns: [eval, exec, __import__, ...]
```

## 🛡️ Security Features

- **Code Validation**: Multi-stage validation for syntax, security, and performance
- **Sandboxed Execution**: Isolated execution environment with resource limits
- **Import Restrictions**: Whitelist-based module import control
- **Pattern Scanning**: Detection of malicious code patterns
- **Resource Monitoring**: CPU, memory, and execution time limits

## 📈 Supported Analysis Types

### Statistical Analysis
- Descriptive statistics and data profiling
- Correlation analysis and feature relationships
- Distribution analysis and normality testing
- Hypothesis testing and significance analysis

### Machine Learning
- Customer segmentation (K-means, RFM analysis)
- Sales forecasting (Prophet, time series analysis)
- Anomaly detection algorithms
- Classification and regression models

### Business Intelligence
- Cohort analysis and retention metrics
- Market basket analysis
- Customer lifetime value calculation
- A/B testing and statistical significance

## 📝 CLI Commands

- `help` - Show help and examples
- `history` - Show conversation history
- `clear` - Clear screen
- `new` - Start new session
- `exit/quit` - Exit application

## 🔌 API Integration

Future REST or GraphQL endpoints can import the `AnalysisController`
from `application.controllers`. The controller exposes the same
`start_session`, `analyze_query`, and `get_session_history` methods used by
the CLI, enabling web services to reuse the `AnalysisWorkflow` without
duplicating business logic.

## 🔍 Debugging

Enable debug mode:
```bash
python main.py --debug
```
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses Google BigQuery public datasets
- Powered by Google Gemini AI models
