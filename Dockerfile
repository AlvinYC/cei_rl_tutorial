#FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
MAINTAINER alvin

# account info (pwd is not necessary for this config)
ARG user=docker
#ARG pwd=1234
#this script will create dir to  /home/{$user}/${worddir} like /home/docker/git_repository
#ARG workdir=workspace
#some large package should copy to /home/${user}/$(local_package}
ARG local_package=utils_thisbuild
ARG github=cei_rl_tutorial
#vscode server 1.54.2
ARG vscommit=fd6f3bce6709b121a895d042d343d71f317d74e7

# udpate timezone
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

# install necessary ubuntu application
RUN apt-get update && apt-get install -y \
    apt-utils sudo vim zsh curl git make unzip \
    wget openssh-server rsync iproute2\
    powerline fonts-powerline \
    # necessary ubuntu package for sudo add-apt-repository ppa:deadsnakes/ppa
    software-properties-common \
    # zsh by ssh issue : icons.zsh:168: character not in range
    language-pack-en \
    libsndfile1 \
    unrar

# install https://github.com/openai/gym mention package
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev \
    libosmesa6-dev xvfb ffmpeg curl patchelf \
    libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig

# downgrad conda python version from 3.8.8 -> 3.6.9
RUN conda update conda;\
    conda install --revision 0;\
    conda install python=3.6.9

# docker account
RUN useradd -m ${user} && echo "${user}:${user}" | chpasswd && adduser ${user} sudo;\
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers;\
    chmod 777 /etc/ssh/sshd_config; echo 'GatewayPorts yes' >> /etc/ssh/sshd_config; chmod 644 /etc/ssh/sshd_config

# change workspace
USER ${user}
WORKDIR /home/${user}

# oh-my-zsh setup
ARG omzthemesetup="POWERLEVEL9K_MODE=\"nerdfont-complete\"\n\
ZSH_THEME=\"powerlevel9k\/powerlevel9k\"\n\n\
POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(ip pyenv virtualenv context dir vcs)\n\
POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(status root_indicator background_jobs history time)\n\
POWERLEVEL9K_VIRTUALENV_BACKGROUND=\"green\"\n\
POWERLEVEL9K_PYENV_PROMPT_ALWAYS_SHOW=true\n\
POWERLEVEL9K_PYENV_BACKGROUND=\"orange1\"\n\
POWERLEVEL9K_DIR_HOME_SUBFOLDER_FOREGROUND=\"white\"\n\
POWERLEVEL9K_PYTHON_ICON=\"\\U1F40D\"\n"


# ssh/zsh plugin
RUN cd ~/ ; mkdir .ssh ;\
    sudo mkdir /var/run/sshd ;\
    sudo sed -ri 's/session required pam_loginuid.so/#session required pam_loginuid.so/g' /etc/pam.d/sshd ;\
    sudo ssh-keygen -A ;\
    wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true ;\
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/plugins/zsh-syntax-highlighting ;\
    git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k ;\
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions ;\
    git clone https://github.com/davidparsson/zsh-pyenv-lazy.git ~/.oh-my-zsh/custom/plugins/pyenv-lazy ;\
    echo "source ~/.oh-my-zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc ;\
    sed -i -r "1s/^/export TERM=\"xterm-256color\"\n/" ~/.zshrc ;\
    sed -i -r "2s/^/LC_ALL=\"en_US.UTF-8\"\n/" ~/.zshrc ;\
    sed -i -r "s/^plugins=.*/plugins=(git zsh-autosuggestions virtualenv screen pyenv-lazy)/" ~/.zshrc ;\
    sed -i -r "s/^ZSH_THEM.*/${omzthemesetup}/" ~/.zshrc ;\
    wget https://github.com/ryanoasis/nerd-fonts/releases/download/v2.1.0/SourceCodePro.zip ;\
    unzip SourceCodePro.zip -d ~/.fonts ;\
    fc-cache -fv  ;\
    sudo chsh -s $(which zsh) ${user}

#RUN sudo apt-get -y install ipython ipython-notebook 
RUN sudo apt-get -y install ipython

RUN python3 -m pip install --user pip==21.0.1;\
    python3 -m pip install --user ipython==7.16.1;\
    python3 -m pip install --user Flask==1.1.2;\
    python3 -m pip install --user opencc-python-reimplemented==0.1.6;\
    python3 -m pip install --user zhon==1.1.5;\
    python3 -m pip install --user pycnnum==1.0.1;\
    python3 -m pip install --user gdown==3.12.2;\
    #python3 -m pip install --user tensorflow-gpu==1.14.0;\
    # NTU 2021 RL sample code package
    python3 -m pip install --user numpy==1.19.5;\
    python3 -m pip install --user torch==1.8.1;\
    python3 -m pip install --user 'gym[box2d]'==0.18.3;\
    python3 -m pip install --user pyvirtualdisplay==2.2;\
    python3 -m pip install --user opencv-python==4.5.3.56;\
    python3 -m pip install --user gym==0.18.3;\
    python3 -m pip install --user 'gym[atari]';\
    #python3 -m pip install --user scipy==1.2.1;\
    python3 -m pip install --user tqdm==4.61.2;\

    #python3 -m pip install --user http://download.pytorch.org/whl/cu91/torch-0.3.1-cp36-cp36m-linux_x86_64.whl;\
    python3 -m pip install --user lws==1.2.7;\
    python3 -m pip install --user unidecode==1.2.0;\
    python3 -m pip install --user inflect==5.3.0;\
    python3 -m pip install --user nnmnkwii==0.0.22;\
    python3 -m pip install --user tensorboardX==2.2;\
    python3 -m pip install --user nltk==3.5;\
    python3 -m pip install --user jupyter==1.0.0;\
    python3 -m pip install --user librosa==0.8.0;\
    python3 -m pip install --user matplotlib==3.3.4;\
    python3 -m pip install --user docopt==0.6.2;\
    # project git clone
    git clone https://github.com/AlvinYC/${github}.git /home/${user}/${github};
    # fix pycnnum issue, ref: https://github.com/zcold/pycnnum/issues/4
    #sed -ir 's/return \[system\.digits\[0.*/return \[system.digits\[0\], system.digits\[int\(striped_string\)\]\]/' \
    #/home/${user}/.local/lib/python3.6/site-packages/pycnnum/pycnnum.py
    # fix tensorboardX issue, add_image default dataformats information from CHW -> HWC
    #sed -ir "s/= 'CHW')/= 'HWC')/" /home/${user}/.local/lib/python3.6/site-packages/tensorboardX/writer.py;\
    # fix vctk 0.92 format issue 
    #sed -Ei "s/(assert len\(fields\).*)/#\1\n            if len(fields)>6: continue/ " /home/docker/.local/lib/python3.6/site-packages/nnmnkwii/datasets/vctk.py


# remote plot matplotlib output (Mac --> dockerhost --> container)
RUN sudo apt-get install libcairo2-dev pkg-config python3-dev libgirepository1.0-dev -y;\
    sudo apt-get install python3-gi gobject-introspection gir1.2-gtk-3.0 xauth -y;\
    sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module -y;\
    python3 -m pip install --user pycairo==1.19.1 --no-use-pep517;\
    python3 -m pip install --user gobject==0.1.0 PyGObject==3.30.5 --no-use-pep517;\
    sudo sed -iE "s/X11Forwarding yes/X11UseLocalhost no\nX11Forwarding yes/" /etc/ssh/sshd_config;\
    echo "export LANG=en_US.UTF-8" >> /home/${user}/.zshrc;\
    echo "export LANGUAGE=en_US:en" >> /home/${user}/.zshrc;\
    echo "export LC_ALL=en_US.UTF-8" >> /home/${user}/.zshrc

# run ./utils_thisbuild/project_setup.sh
COPY ${local_package} /home/${user}/${local_package}
RUN  sudo chown -R ${user}:${user} /home/${user}/${local_package};\
     echo "alias watch1=watch -n 0.5" >> ~/.zshrc;\
     echo "export PATH=/home/${user}/.local/bin:$PATH" >> ~/.zshrc;\
     sh /home/${user}/${local_package}/project_setup.sh

# vscode server part
RUN curl -sSL "https://update.code.visualstudio.com/commit:${vscommit}/server-linux-x64/stable" -o /home/${user}/${local_package}/vscode-server-linux-x64.tar.gz;\
    mkdir -p ~/.vscode-server/bin/${vscommit};\
    tar zxvf /home/${user}/${local_package}/vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${vscommit} --strip 1;\
    touch ~/.vscode-server/bin/${vscommit}/0

# jupyter notebook config 'auto newline in cell'
ARG JUCELL="{\
  \"MarkdownCell\": {\
    \"cm_config\": {\
      \"lineWrapping\": true\
    }\
  },\
  \"CodeCell\": {\
    \"cm_config\": {\
      \"lineWrapping\": true\
    }\
  }\
}"

# update jupyter notebook config 
RUN /home/${user}/.local/bin/jupyter notebook --generate-config;\
    sed -ir "s/\#c\.NotebookApp\.token.*/c\.NotebookApp\.token = \'\'/" ~/.jupyter/jupyter_notebook_config.py;\
    sed -ir "s/#c\.NotebookApp\.password =.*/c\.NotebookApp\.password = u\'\'/" ~/.jupyter/jupyter_notebook_config.py;\
    sed -ir "s/#c\.NotebookApp\.ip = .*/c\.NotebookApp\.ip = \'\*\'/" ~/.jupyter/jupyter_notebook_config.py;\
    sed -ir "s/#c\.NotebookApp\.notebook_dir.*/c\.NotebookApp\.notebook_dir = \'\/home\/docker\/${github}\'/" ~/.jupyter/jupyter_notebook_config.py;\
    mkdir -p ~/.jupyter/nbconfig;\
    echo ${JUCELL} > ~/.jupyter/nbconfig/notebook.json        
 
ADD id_rsa.pub /home/${user}/.ssh/authorized_keys

ENTRYPOINT sudo service ssh restart && zsh
                    

