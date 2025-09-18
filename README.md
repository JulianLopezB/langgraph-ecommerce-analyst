# LangGraph Data Analysis Agent

An AI-powered data analysis agent that processes e-commerce data from Google BigQuery and generates actionable business insights through natural language interactions.

> **Recent Updates**: Enhanced DevOps strategy with Docker Compose orchestration, centralized dataset configuration management, and improved workflow architecture for better scalability and maintainability.

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
   git clone <your-repository-url>
   cd <your-project-directory>
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
   
   # BigQuery Configuration (optional - uses centralized dataset config)
   BQ_DATASET_ID=bigquery-public-data.thelook_ecommerce

   # Deployment Environment (optional: development|production)
   APP_ENV=development
   
   # LangSmith Tracing (optional - for debugging)
   LANGSMITH_API_KEY=your_langsmith_api_key_here

   # OpenTelemetry (optional - tracing backend configuration)
   OTEL_EXPORTER_JAEGER_AGENT_HOST=localhost
   OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
   # Optional OTLP endpoint (e.g., OpenSearch or collector)
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
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

## 🐳 DevOps & Deployment

### Docker Compose Strategy

The project implements a comprehensive DevOps strategy with Docker Compose for local development and testing, providing a complete containerized environment that mirrors production setups.

Run the application alongside its dependencies using Docker Compose.

1. Ensure a `.env` file exists in the project root with the required configuration values.
2. Build and start all services:
   ```bash
   docker-compose up --build
   ```
   This launches the application, a PostgreSQL database, and a RabbitMQ broker on an isolated
   `app-network`, persists data in the `db_data` and `rabbitmq_data` volumes, and allocates a TTY
   for the app service so you can use the interactive CLI directly.
3. Stop the services when finished:
   ```bash
   docker-compose down
   ```

**DevOps Features:**
- **Service Orchestration**: Coordinated startup of application, database, and messaging services
- **Network Isolation**: Services communicate through a dedicated `app-network`
- **Data Persistence**: Persistent volumes for `db_data` and `rabbitmq_data`
- **Development Workflow**: TTY allocation for interactive CLI usage during development
- **Environment Consistency**: Identical configuration between development and production environments

## 🐳 Docker

The Docker image installs system build tools and pre-builds CmdStan so Prophet-based
forecasting works out of the box.

Build the image:

```bash
 docker build -t langgraph-data-analyst .
```

Run the container:

```bash
 docker run --rm -it -e GEMINI_API_KEY=your_gemini_api_key_here langgraph-data-analyst
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

The system follows a modular, layered architecture with recent improvements focusing on workflow orchestration and centralized configuration management:

### Core Components

- **LangGraph Workflow**: Orchestrates the analysis workflow
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

### Centralized Dataset Configuration

The system uses a centralized configuration approach for managing BigQuery datasets and connection settings. This ensures consistent data access patterns across all components and simplifies environment management.

**Key Features:**
- **Default Dataset**: Uses the public `bigquery-public-data.thelook_ecommerce` dataset for demonstrations
- **Custom Projects**: Easily configurable for your own BigQuery projects
- **Environment-Specific Settings**: Different configurations for development and production environments
- **Connection Pooling**: Optimized connection management for better performance

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

## 📈 Observability

The agent emits OpenTelemetry traces and JSON-formatted logs.

### Tracing

1. Ensure a Jaeger instance or OpenTelemetry collector is running.
2. Set `OTEL_EXPORTER_JAEGER_AGENT_HOST` and related environment variables.
3. Start the application and open [http://localhost:16686](http://localhost:16686) to explore traces in Jaeger.
4. If using an OTLP endpoint (e.g., OpenSearch), view traces using your collector's dashboard.

### Logs

- Structured logs are written to `logs/agent.log` in JSON format.
- Forward this file to your log aggregator (e.g., OpenSearch) for centralized analysis.

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
