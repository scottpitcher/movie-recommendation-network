movie-recommendation-network/
│
├── data/                          # Data directory
│   ├── raw_data/                  # Raw data files
│   │   └── full_data_with_credits.csv  # Raw data with credits from TMDB API
│   ├── processed_data/            # Processed data files for Neo4j and ML pipeline
│   │   ├── full_data_binned_log.csv  # Log-transformed and binned data
│   │   └── full_data_nodes.csv    # Data with node degrees added
│   └── training_data/             # Final training-ready datasets
│       └── shared_matrix.npy      # 3D shared feature matrix
│
├── models/                        # Directory for saved models
│   └── trained_model.pt           # Example PyTorch trained model
│
├── scripts/                       # Pipeline scripts for automation
│   ├── api_fetch.py               # Fetch raw data from TMDB -> raw_data/full_data_with_credits.csv
│   ├── data_preprocessing.py      # Preprocess raw data -> processed_data/full_data_binned_log.csv
│   ├── data_ingestion.py          # Ingest processed data into Neo4j -> Prepares graph DB
│   ├── shared_features.py         # Generate shared feature matrix from Neo4j -> processed_data/shared_matrix.npy
│   ├── data_processing.py         # Process final dataset for model training -> training_data/final_processed_data.csv
│   ├── feature_engineering.py     # Prepare final feature set for model input
│   ├── model_architecture.py      # Define PyTorch model architecture
│   ├── model_training.py          # Train model on generated feature sets
│   ├── model_evaluation.py        # Evaluate model performance
│   ├── model_deployment.py        # Deployment utilities for serving the trained model
│   └── logging_utils.py           # Centralized logging utilities for all scripts
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── data-processing.ipynb      # Interactive exploration of data preprocessing
│   └── model-training.ipynb       # Interactive exploration of model training
│
├── Dockerfile                     # For containerization of the project
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── pipeline.py                    # Central script to orchestrate all steps
