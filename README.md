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

3. **Configure secrets**:
   Store required API keys in AWS Secrets Manager and expose their names via environment variables:
   ```bash
   export GEMINI_SECRET_NAME=GEMINI_API_KEY
   export LANGCHAIN_SECRET_NAME=LANGCHAIN_API_KEY
   export AWS_REGION=your_aws_region
   ```

4. **Set up Google Cloud authentication** (if using your own project):
   ```bash
   gcloud auth application-default login
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

## 🔐 Secret Management

- Secrets are stored in AWS Secrets Manager.
- Rotate API keys every 90 days or upon potential exposure.
- Grant application IAM roles only the `secretsmanager:GetSecretValue` permission for required secrets.

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
