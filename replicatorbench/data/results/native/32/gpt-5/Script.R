# Entrypoint Script.R to run replication analysis inside container
# It sources the container-ready analysis script located under /app/data

source("/app/data/replication_analysis.R")
