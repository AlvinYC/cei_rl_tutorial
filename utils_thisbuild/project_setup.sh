#cd ~/
#sh -c 'mv utils_thisbuild content'
cd ~/utils_thisbuild

# download atari ROM 
wget http://www.atarimania.com/roms/Roms.rar 
unrar x Roms.rar
unzip 'HC ROMS.zip'
cd 'HC ROMS'
python3 -m atari_py.import_roms .\

