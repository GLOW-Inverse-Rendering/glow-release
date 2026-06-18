FROM wujiaye1996/pytorch:2.4.1
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update \
    && apt-get install -y git ninja-build curl ninja-build
RUN --mount=type=bind,source=patches,target=/patches cd / && git clone --recurse-submodules https://github.com/19reborn/NeuS2_TCNN.git \
    && cd NeuS2_TCNN \
    && patch -u bindings/torch/setup.py /patches/neus2_tcnn.patch  \
    && TCNN_CUDA_ARCHITECTURES="75"  pip install ./bindings/torch && cd / && rm -rf tiny-cuda-nn
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update \
    && apt-get install -y libjpeg-dev ffmpeg libsm6 libxext6 

# RUN --mount=type=bind,source=mitsuba3_output,target=/mitsuba3_output \
#     cd /mitsuba3_output && pip3 install --force-reinstall ./*.whl
    

RUN  pip install --no-cache-dir  nerfstudio==1.1.4
RUN --mount=type=bind,source=requirements.txt,target=/requirements.txt --mount=type=bind,source=mitsuba3_output,target=/mitsuba3_output \
    cd / && TORCH_CUDA_ARCH_LIST="7.5 8.6" FORCE_CUDA=1 pip install --no-cache-dir -r /requirements.txt  "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    cd /mitsuba3_output && pip3 install --force-reinstall ./*.whl
# RUN pip install  --no-cache-dir -U --force-reinstall Pillow-SIMD==9.5.0.post2

# RUN TORCH_CUDA_ARCH_LIST="7.5 8.6" FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
