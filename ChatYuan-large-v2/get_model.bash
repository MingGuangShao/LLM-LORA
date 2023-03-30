
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs

cd ./model/

git lfs install
git clone https://huggingface.co/ClueAI/ChatYuan-large-v2
