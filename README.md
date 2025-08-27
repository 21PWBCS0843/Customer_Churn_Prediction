# Customer Churn Prediction System

A comprehensive web-based machine learning application that predicts customer churn using advanced ensemble techniques and provides actionable business insights through an intuitive dashboard interface.

## Project Overview

This system leverages machine learning to identify customers at risk of churning and provides businesses with data-driven insights for proactive customer retention strategies. Built as a final year project for the Department of Computer Science & IT, University of Engineering & Technology Peshawar.

### Key Features

- **High Accuracy Predictions**: Achieves 86.5% accuracy using ensemble learning
- **Individual & Batch Processing**: Support for single customer analysis and bulk predictions
- **Customer Segmentation**: K-means clustering identifies distinct customer groups
- **Interactive Dashboard**: Real-time analytics with comprehensive visualizations
- **PDF Report Generation**: Professional reports with actionable recommendations
- **Responsive Web Interface**: Cross-browser compatible with mobile support

## 🚀 Technical Stack

### Backend
- **Framework**: Flask (Python)
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Report Generation**: ReportLab

### Frontend
- **Languages**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js
- **Design**: Responsive grid-based layout

### Machine Learning Models
- **Logistic Regression**: Baseline predictions with interpretable coefficients
- **Random Forest**: Non-linear pattern detection with feature importance
- **Gradient Boosting**: Sequential learning for error refinement
- **Ensemble Method**: Weighted voting system combining all models

## 📊 System Architecture

The system follows a three-tier architecture:

1. **Presentation Layer**: Web interface with interactive dashboards
2. **Application Layer**: Flask API with ML pipeline and business logic
3. **Data Layer**: File-based data management with model artifacts

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/21PWBCS0843/Customer_Churn_Prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`

## 📋 Usage Guide

### Individual Customer Prediction
1. Navigate to the "Single Customer Prediction" page
2. Enter customer details (demographics, financial information, account data)
3. Submit the form to get:
   - Churn probability score
   - Risk level categorization
   - Customer segment classification
   - Personalized retention recommendations

### Batch Processing
1. Go to "Batch Customer Prediction" section
2. Download the CSV template or check required fields
3. Upload your customer data file (CSV format)
4. Process up to 10,000 records simultaneously
5. Download comprehensive analysis reports

### Dashboard Analytics
- View real-time churn statistics and KPIs
- Explore customer risk distribution
- Analyze feature importance rankings
- Monitor churn trends across segments

## 📈 Model Performance

| Metric | Individual Models | Ensemble |
|--------|------------------|----------|
| Logistic Regression | 82.3% | - |
| Random Forest | 84.7% | - |
| Gradient Boosting | 86.1% | - |
| **Final Ensemble** | - | **86.5%** |
| Precision | - | 83.4% |
| Recall | - | 86.1% |
| F1-Score | - | 84.7% |

## 🎨 Features in Detail

### Advanced Feature Engineering
- **Ratio-based Features**: Balance-to-Salary, Credit-to-Salary ratios
- **Behavioral Indicators**: Products-per-tenure, activity flags
- **Interaction Features**: Young inactive customers, multiple product holders

### Customer Segmentation
The system identifies four distinct customer segments:
- **Young Professionals**: Early career customers with growth potential
- **Established Savers**: Stable, long-term customers with high balances
- **High-Value Clients**: Premium customers requiring specialized retention
- **At-Risk Seniors**: Older customers with higher churn probability

### Data Preprocessing Pipeline
- Missing value imputation (median for numerical, mode for categorical)
- Outlier detection using IQR method with capping (not removal)
- Data validation and integrity checks
- Feature scaling and normalization

## 🔍 API Endpoints

### Individual Prediction
```http
POST /predict
Content-Type: application/json

{
  "age": 35,
  "credit_score": 650,
  "balance": 50000,
  "estimated_salary": 75000,
  "tenure": 3,
  "num_of_products": 2,
  "has_cr_card": 1,
  "is_active_member": 1,
  "geography": "France",
  "gender": "Male"
}
```

### Batch Processing
```http
POST /batch_predict
Content-Type: multipart/form-data

file: customer_data.csv
```

### Dashboard Data
```http
GET /dashboard_data
```

## 📊 Performance Metrics

### System Performance
- **Individual Predictions**: < 2 seconds response time
- **Batch Processing**: 1,200+ predictions per minute
- **Dashboard Loading**: < 3 seconds
- **Concurrent Users**: Supports up to 50 simultaneous users

### Browser Compatibility
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Android Chrome)

## 🧪 Testing

The system has undergone comprehensive testing:

### Functional Testing
- ✅ All API endpoints tested and validated
- ✅ Form validation and error handling
- ✅ Cross-browser compatibility verified
- ✅ Mobile responsiveness confirmed

### Performance Testing
- ✅ Load testing up to 50 concurrent users
- ✅ Response times within specifications
- ✅ Memory usage optimization validated

### Model Validation
- ✅ Cross-validation with 5-fold methodology
- ✅ Performance metrics exceed targets
- ✅ Prediction consistency across segments

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                 # Main Flask application
├── models/               # ML model artifacts
├── templates/            # HTML templates
├── static/              
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Static images
├── data/                # Data processing utilities
├── reports/             # Generated PDF reports
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚧 Future Enhancements

### Technical Roadmap
- **Real-time Data Streaming**: Continuous model updates
- **Deep Learning Integration**: Advanced temporal pattern detection
- **AutoML Implementation**: Automated model optimization
- **Cloud Deployment**: Containerization with Docker/Kubernetes

### Business Features
- **CRM Integration**: Seamless integration with existing systems
- **A/B Testing Framework**: Retention strategy optimization
- **Advanced Analytics**: Causal inference and explainable AI
- **Multi-tenant Architecture**: Support for multiple organizations

## 👥 Contributors

- **Haseeb Hassan** (21PWBCS0843) - Lead Developer & ML Engineer
- **Uzma Masroor** (21PWBCS0830) - Frontend Developer & Data Analyst

**Supervisor**: Ma'am Ayesha Javid  
**Institution**: University of Engineering & Technology Peshawar

## 📄 License

This project is developed as part of academic requirements for Bachelor's degree in Computer Science. All rights reserved.

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome. Please feel free to:
- Report bugs or issues
- Suggest feature improvements
- Share usage experiences

## 📞 Support

For questions or support regarding this project:
- Create an issue in this repository
- Contact the development team through the university

## 🏆 Acknowledgments

- **Allah Almighty** for guidance and knowledge
- **Ma'am Ayesha Javid** for supervision and support
- **UET Peshawar** faculty for educational foundation
- Open source community for tools and libraries

---

**Note**: This system is designed for educational and research purposes. For production deployment, additional security measures and scalability considerations should be implemented.
