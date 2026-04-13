# 1. Use a lightweight Python 3.13 base image
FROM python:3.13-slim

# 2. Install 'uv' directly from the official source
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy ONLY the dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock ./

# 5. Install dependencies using uv (fast and strict)
RUN uv sync --frozen --no-dev

# 6. Copy the rest of your application code
COPY . .

# 7. Expose the port Streamlit uses
EXPOSE 8501

# 8. Command to run the application
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"] 