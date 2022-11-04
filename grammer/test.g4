grammar test;		
prog:	expr EOF ;
expr:	expr ('*'|'/') expr
    |	expr ('+'|'-') expr
    |	'(' expr ')'
    |   VARIABLE
    ;
ITEM: NAME;
VARIABLE: '$' .+;
NEWLINE : [\r\n]+ -> skip;
INT     : [0-9]+ ;
NAME    : ('a' .. 'z' | 'A' .. 'Z' | '\u4E00'..'\u9FA5' | '0' .. '9' | '_')+ ;
// antlr4-parse test.g4 prog -gui
// antlr4 -Dlanguage=Python3 test.g4