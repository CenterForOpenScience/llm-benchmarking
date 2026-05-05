FROM rocker/verse:4.3.1

# System deps and R base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget ca-certificates make r-base r-base-core r-base-dev \
  && rm -rf /var/lib/apt/lists/*

# Ensure R is available
RUN command -v R || (apt-get update && apt-get install -y r-base)

# Install required R packages
RUN R -q -e 'install.packages(c("readstata13","tidyverse","reshape2","prais","panelAR","foreign"), repos="https://cloud.r-project.org")'

# Create a robust Rscript wrapper in /usr/local/bin only
RUN sh -c 'printf "#!/usr/bin/env bash\nset -e\nif [ \"$#\" -lt 1 ]; then echo \"Usage: Rscript <file> [args...]\"; exit 1; fi\nfile=\"$1\"; shift || true\nexec R -q -f \"$file\" --args \"$@\"\n" > /usr/local/bin/Rscript && chmod +x /usr/local/bin/Rscript'

WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]