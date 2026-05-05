FROM rocker/r-ver:4.3.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git wget build-essential gfortran libnlopt-dev liblapack-dev libblas-dev libcurl4-openssl-dev libxml2-dev libssl-dev r-cran-lme4 r-cran-mice r-cran-tidyverse && rm -rf /var/lib/apt/lists/*
RUN command -v R || (apt-get update && apt-get install -y r-base)
RUN R -q -e 'install.packages(c("lme4","mice","tidyverse"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]