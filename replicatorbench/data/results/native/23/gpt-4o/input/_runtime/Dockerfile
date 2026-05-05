FROM rocker/verse
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN command -v R || (apt-get update && apt-get install -y r-base)
RUN R -q -e 'install.packages(c("readxl","readr","tidyverse","ggplot2","kableExtra","pander","dplyr","emmeans","sjstats","tibble","reshape2","ggpubr"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp /app/replication_data && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]