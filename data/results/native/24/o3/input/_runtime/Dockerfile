FROM rocker/verse:4.3.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y pandoc && rm -rf /var/lib/apt/lists/*
RUN command -v R || (apt-get update && apt-get install -y r-base)
RUN R -q -e 'install.packages(c("tidyverse (>=1.3.2)","jsonlite","papaja (>=0.1.1)","broom","ggplot2"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]