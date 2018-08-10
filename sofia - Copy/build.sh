pip3 uninstall smarter || true
python3 setup.py sdist
pip3 install dist/smarter-0.1.2.tar.gz