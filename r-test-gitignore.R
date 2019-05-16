library(dotenv)

getwd()
setwd('/Users/dengruo/Git/Anomly-Detection-SARIMA')

load_dot_env('.env')

user = Sys.getenv("SECRET_KEY")
