FROM rocker/verse:4.3.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
# Ensure Rscript is available; create a shim if not present and place in common PATH
RUN /bin/sh -lc 'if ! command -v Rscript >/dev/null 2>&1; then printf "#!/bin/sh\nexec R -q -f \"$1\"\n" > /usr/local/bin/Rscript && chmod +x /usr/local/bin/Rscript && cp /usr/local/bin/Rscript /usr/bin/Rscript; fi'
# Install required R packages (CRAN and Bioconductor)
RUN R -q -e 'install.packages(c("lme4","lmerTest","afex","car","dplyr","plyr","metaRNASeq","BiocManager","metap"), repos="https://cloud.r-project.org"); if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install("multtest", ask=FALSE, update=FALSE)'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
ENV PATH="/usr/local/bin:/usr/bin:/bin"
USER runner
CMD ["bash"]