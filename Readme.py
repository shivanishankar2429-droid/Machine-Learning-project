
"""
Installation & Setup

1 Navigate to the project directory
cd /Users/lalith/Desktop/machinelearning

2 Install pyenv (if not already installed)
# For Homebrew users
brew install pyenv


3 Initialize pyenv in your shell:

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
exec "$SHELL"


4 Verify: pyenv --version

5 Install Python 3.10
pyenv install 3.10.13
pyenv global 3.10.13
python --version

6 Create and activate a virtual environment
python -m venv venv
source venv/bin/activate


7 Verify:python --version pip --version

8 Upgrade pip
pip install --upgrade pip

9 Install required Python packages
# Remove incompatible packages if any
pip uninstall -y numpy scikit-learn imbalanced-learn
pip cache purge

# Install compatible versions
pip install numpy==1.23.5
pip install scikit-learn==1.2.2 imbalanced-learn==0.11.0

# Install other dependencies
pip install streamlit pandas matplotlib seaborn plotly shap

10 Verify installation
python - <<EOF
from imblearn.over_sampling import SMOTE
import numpy, sklearn, imblearn
print("All packages installed correctly!")
EOF

11. Run the Streamlit App
streamlit run app2.py


12 After execution, the app will open automatically in your browser at:
http://localhost:8501

"""