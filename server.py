import dash

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=['css/bootstrap.min.css', 'css/style.css']
)

# set web title
app.title = 'ARLight'

server = app.server
