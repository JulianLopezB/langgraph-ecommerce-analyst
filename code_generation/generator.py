"""Main code generation orchestrator for dynamic Python and SQL generation."""
import logging
from typing import Dict, Any, Optional
from enum import Enum

from agent.state import IntentType, GeneratedCode
from code_generation.templates.statistical_templates import (
    DESCRIPTIVE_STATS_TEMPLATE,
    CORRELATION_ANALYSIS_TEMPLATE,
    DISTRIBUTION_ANALYSIS_TEMPLATE
)
from code_generation.templates.ml_templates import (
    CUSTOMER_SEGMENTATION_TEMPLATE,
    FORECASTING_TEMPLATE
)

logger = logging.getLogger(__name__)


class CodeGenerationStrategy(Enum):
    """Code generation strategies."""
    TEMPLATE_BASED = "template_based"
    LLM_GENERATED = "llm_generated"
    HYBRID = "hybrid"


class CodeGenerator:
    """Main code generation orchestrator."""
    
    def __init__(self):
        """Initialize code generator with templates."""
        self.templates = {
            # Statistical Analysis Templates
            IntentType.DATA_EXPLORATION: {
                "descriptive_stats": DESCRIPTIVE_STATS_TEMPLATE,
                "correlation": CORRELATION_ANALYSIS_TEMPLATE,
                "distribution": DISTRIBUTION_ANALYSIS_TEMPLATE
            },
            
            # Customer Analysis Templates
            IntentType.CUSTOMER_ANALYSIS: {
                "segmentation": CUSTOMER_SEGMENTATION_TEMPLATE,
                "rfm": CUSTOMER_SEGMENTATION_TEMPLATE  # Same template
            },
            
            # Sales Analysis Templates
            IntentType.SALES_ANALYSIS: {
                "forecasting": FORECASTING_TEMPLATE,
                "trends": FORECASTING_TEMPLATE
            },
            
            # Advanced Analytics
            IntentType.ADVANCED_ANALYTICS: {
                "forecasting": FORECASTING_TEMPLATE,
                "segmentation": CUSTOMER_SEGMENTATION_TEMPLATE
            }
        }
        
        # Template selection keywords
        self.template_keywords = {
            "descriptive": "descriptive_stats",
            "correlation": "correlation",
            "distribution": "distribution",
            "segment": "segmentation",
            "rfm": "rfm",
            "forecast": "forecasting",
            "trend": "trends",
            "customer": "segmentation",
            "churn": "segmentation",
            "clv": "segmentation",
            "sales": "forecasting",
            "revenue": "forecasting",
            "predict": "forecasting"
        }
    
    def generate_code(
        self, 
        intent: IntentType, 
        intent_data: Dict[str, Any], 
        data_info: Dict[str, Any],
        strategy: CodeGenerationStrategy = CodeGenerationStrategy.TEMPLATE_BASED
    ) -> GeneratedCode:
        """
        Generate Python code based on intent and data information.
        
        Args:
            intent: Classified user intent
            intent_data: Detailed intent information from LLM
            data_info: Information about the dataset
            strategy: Code generation strategy to use
            
        Returns:
            GeneratedCode object with generated code
        """
        logger.info(f"Generating code for intent: {intent} using strategy: {strategy}")
        
        try:
            if strategy == CodeGenerationStrategy.TEMPLATE_BASED:
                return self._generate_template_based(intent, intent_data, data_info)
            elif strategy == CodeGenerationStrategy.LLM_GENERATED:
                # This would use the LLM service directly
                # For now, fall back to template-based
                logger.warning("LLM_GENERATED strategy not implemented, falling back to template-based")
                return self._generate_template_based(intent, intent_data, data_info)
            else:  # HYBRID
                # Try template first, fall back to LLM if needed
                return self._generate_template_based(intent, intent_data, data_info)
                
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            raise
    
    def _generate_template_based(
        self, 
        intent: IntentType, 
        intent_data: Dict[str, Any], 
        data_info: Dict[str, Any]
    ) -> GeneratedCode:
        """Generate code using predefined templates."""
        
        # Select appropriate template
        template_name = self._select_template(intent, intent_data)
        
        if intent not in self.templates or template_name not in self.templates[intent]:
            # Fall back to descriptive stats for unknown intents
            logger.warning(f"No template found for {intent}/{template_name}, using descriptive stats")
            template_code = DESCRIPTIVE_STATS_TEMPLATE
            template_name = "descriptive_stats"
        else:
            template_code = self.templates[intent][template_name]
        
        # Apply template customizations based on data
        customized_code = self._customize_template(template_code, intent_data, data_info)
        
        return GeneratedCode(
            code_content=customized_code,
            template_used=f"{intent.value}_{template_name}",
            parameters={
                "intent": intent_data,
                "data_info": data_info,
                "template_name": template_name
            }
        )
    
    def _select_template(self, intent: IntentType, intent_data: Dict[str, Any]) -> str:
        """Select the most appropriate template based on intent and keywords."""
        
        query_text = intent_data.get("analysis_type", "").lower()
        
        # Check for specific keywords in the analysis type
        for keyword, template_name in self.template_keywords.items():
            if keyword in query_text:
                logger.info(f"Selected template '{template_name}' based on keyword '{keyword}'")
                return template_name
        
        # Default templates by intent type
        defaults = {
            IntentType.DATA_EXPLORATION: "descriptive_stats",
            IntentType.CUSTOMER_ANALYSIS: "segmentation", 
            IntentType.SALES_ANALYSIS: "forecasting",
            IntentType.PRODUCT_ANALYSIS: "descriptive_stats",
            IntentType.ADVANCED_ANALYTICS: "segmentation",
            IntentType.VISUALIZATION: "descriptive_stats"
        }
        
        template_name = defaults.get(intent, "descriptive_stats")
        logger.info(f"Using default template '{template_name}' for intent {intent}")
        return template_name
    
    def _customize_template(
        self, 
        template_code: str, 
        intent_data: Dict[str, Any], 
        data_info: Dict[str, Any]
    ) -> str:
        """Customize template code based on specific data characteristics."""
        
        customized_code = template_code
        
        # Add data-specific customizations
        columns = data_info.get("columns", [])
        
        # Add column existence checks for critical columns
        critical_columns_check = self._generate_column_checks(columns, intent_data)
        if critical_columns_check:
            customized_code = critical_columns_check + "\\n\\n" + customized_code
        
        # Add data preprocessing if needed
        preprocessing = self._generate_preprocessing(data_info)
        if preprocessing:
            customized_code = preprocessing + "\\n\\n" + customized_code
        
        return customized_code
    
    def _generate_column_checks(self, columns: list, intent_data: Dict[str, Any]) -> str:
        """Generate code to check for required columns."""
        intent = intent_data.get("intent", "")
        
        required_columns = []
        
        # Define required columns by analysis type
        if intent in ["customer_analysis", "segmentation"]:
            required_columns = ["user_id"]
        elif intent in ["sales_analysis", "forecasting"]:
            required_columns = ["created_at", "sale_price"]
        elif intent == "product_analysis":
            required_columns = ["product_id", "sale_price"]
        
        if not required_columns:
            return ""
        
        # Generate column existence check
        check_code = "# Check for required columns\\n"
        for col in required_columns:
            # Look for similar column names (case-insensitive, with variations)
            variations = [col, col.lower(), col.upper(), col.replace("_", "")]
            found_col = None
            
            for variation in variations:
                if variation in columns:
                    found_col = variation
                    break
            
            if found_col:
                check_code += f"# Found required column: {found_col}\\n"
            else:
                check_code += f"""
if '{col}' not in df.columns:
    available_cols = [col for col in df.columns if '{col.split('_')[0]}' in col.lower()]
    if available_cols:
        print(f"Column '{col}' not found. Available similar columns: {available_cols}")
        # Try to use the first similar column
        df = df.rename(columns={{available_cols[0]: '{col}'}})
    else:
        print(f"Warning: Required column '{col}' not found in dataset")
"""
        
        return check_code
    
    def _generate_preprocessing(self, data_info: Dict[str, Any]) -> str:
        """Generate data preprocessing code based on data characteristics."""
        preprocessing_code = "# Data preprocessing\\n"
        
        # Check for missing data
        null_counts = data_info.get("null_counts", {})
        if any(count > 0 for count in null_counts.values()):
            preprocessing_code += """
# Handle missing data
print("Checking for missing data...")
missing_summary = df.isnull().sum()
if missing_summary.sum() > 0:
    print("Missing data found:")
    print(missing_summary[missing_summary > 0])
"""
        
        # Check for date columns that need parsing
        dtypes = data_info.get("dtypes", {})
        date_columns = [col for col, dtype in dtypes.items() 
                       if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            preprocessing_code += f"""
# Convert date columns to datetime
date_columns = {date_columns}
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
"""
        
        return preprocessing_code if preprocessing_code != "# Data preprocessing\\n" else ""
    
    def get_available_templates(self) -> Dict[str, list]:
        """Get list of available templates by intent type."""
        return {
            intent.value: list(templates.keys()) 
            for intent, templates in self.templates.items()
        }
    
    def add_custom_template(self, intent: IntentType, name: str, template_code: str):
        """Add a custom template for a specific intent."""
        if intent not in self.templates:
            self.templates[intent] = {}
        
        self.templates[intent][name] = template_code
        logger.info(f"Added custom template '{name}' for intent {intent}")


# Global code generator instance
code_generator = CodeGenerator()
