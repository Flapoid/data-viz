# Financial Dashboard

A comprehensive financial dashboard built with Streamlit for analyzing fund performance, holdings, and risk metrics.

## Features

- Fund performance comparison and analysis
- Holdings visualization
- Risk metrics analysis
- Style box analysis
- PDF report generation
- Interactive data visualization

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

## Local Development Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd financial-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run src/app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended for personal/small projects)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your app by selecting your repository

### 2. Heroku

1. Install Heroku CLI and login:
```bash
heroku login
```

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Deploy:
```bash
git push heroku main
```

### 3. Docker (For custom deployments)

1. Build the Docker image:
```bash
docker build -t financial-dashboard .
```

2. Run the container:
```bash
docker run -p 8501:8501 financial-dashboard
```

## Environment Variables

The following environment variables can be set:

- `PORT`: Port number (default: 8501)
- Add any other environment variables your app needs

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.