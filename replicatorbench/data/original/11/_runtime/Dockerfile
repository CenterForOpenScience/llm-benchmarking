FROM rocker/verse
RUN R -q -e 'install.packages(c("lmerTest","dplyr","tidyverse","MuMIn","reshape"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]