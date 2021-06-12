import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from views.index import index_page
from views.data import data_page

from server import app

app.layout = html.Div(
    [
        dcc.Location(id='url'),
        dbc.Navbar(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=app.get_asset_url("home.png"), height="60px"),
                                    width={'size': 2, 'offset': 3}),
                            dbc.Col(
                                dbc.NavbarBrand("ARLight", className="ml-1 mt-1",
                                                style={'color': 'black',
                                                       'font-size': '1.6em',
                                                       'font-weight': 'bold',
                                                       'font-family': 'ui-monospace'}),
                                width={'size': 2, 'offset': 2}
                            ),
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="http://ccfcv.ccf.org.cn/ccfcv/wyhdt/dyfc/2018ndwyfc/2020-03-04/695836.shtml",
                    style={'text-decoration': 'none'}
                ),

                dbc.Row(
                    [
                        dbc.Col(dbc.NavLink('Home', href='/', active="exact", className="mt-1",
                                            style={'color': 'black',
                                                   'font-size': '1.2em',
                                                   'font-weight': 'bold',
                                                   'font-family': 'ui-monospace'}
                                            )),
                        dbc.Col(dbc.NavLink('Data', href='/data', active="exact", className="mt-1", id="navlink-data",
                                            style={'color': 'black',
                                                   'font-size': '1.2em',
                                                   'font-weight': 'bold',
                                                   'font-family': 'ui-monospace'}
                                            )
                                )
                    ],
                    align="center",
                    no_gutters=True,
                    className="ml-auto mr-4 flex-nowrap mt-6 md-0",
                )
            ],
            color="#EDEFEB",
            # color="#293a66",
            # dark=True,
            style={'margin-top': '10px'}
        ),

        html.Div(
            id='page-content',
            style={
                'flex': 'auto'
            }
        ),
    ],
    style={

    }
)


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return index_page

    elif pathname == '/data':
        return data_page

    return html.H1('您访问的页面不存在！')


if __name__ == '__main__':
    app.run_server(debug=False)
