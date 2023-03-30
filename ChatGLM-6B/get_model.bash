
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs

cd ./model/

git lfs install
git clone https://huggingface.co/THUDM/chatglm-6b
