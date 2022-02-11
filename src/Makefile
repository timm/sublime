ok:
	mkdir -p ../docs

pdoc: ok
	pdoc -o ../docs sublime.py

pdfs: ../docs/sublime.pdf

../docs/%.pdf : %.py ok
	a2ps  -BjR --line-numbers=1                       \
            --borders=no --pro=color --columns 2 \
            --right-footer="" --left-footer="" \
            --footer="page %p." \
            -M letter -o $@.ps $<
	ps2pdf $@.ps $@
	rm $@.ps
	git add $@


