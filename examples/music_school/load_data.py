import kagglehub

def download():
    path = kagglehub.dataset_download("catherinerasgaitis/mxmh-survey-results")

    print("Path to dataset files:", path)