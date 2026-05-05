FROM rocker/verse:4.4.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y pandoc git && rm -rf /var/lib/apt/lists/*
RUN command -v R || (apt-get update && apt-get install -y r-base)
RUN R -q -e 'install.packages(c("tidyverse","emmeans","kableExtra","Rmisc","car","reshape2","lme4","lmerTest","sjPlot","pwr","rstatix","Matrix","MatrixModels","quantreg","carData","rmarkdown","knitr"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]