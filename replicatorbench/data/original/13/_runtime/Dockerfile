FROM rocker/verse:3.6.3
RUN R -q -e 'install.packages(c("lme4","mice"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]