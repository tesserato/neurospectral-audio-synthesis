pandoc neurospectral.md -o neurospectral.docx ^
--filter pandoc-fignos ^
--filter pandoc-tablenos ^
--filter pandoc-eqnos ^
--bibliography=bib.bib ^
--reference-doc=ref.docx

pause