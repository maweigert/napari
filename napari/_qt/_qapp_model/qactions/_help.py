"""Actions related to the 'Help' menu that require Qt.

'Help' actions that do not require Qt should go in a new '_help_actions.py'
file within `napari/_app_model/actions/`.
"""

import sys
from typing import List

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._qt.dialogs.qt_about import QtAbout
from napari._qt.qt_main_window import Window
from napari.utils.translations import trans

try:
    from napari_error_reporter import ask_opt_in
except ModuleNotFoundError:
    ask_opt_in = None


def _show_about(window: Window):
    QtAbout.showAbout(window._qt_window)


Q_HELP_ACTIONS: List[Action] = [
    Action(
        id=CommandId.NAPARI_INFO,
        title=CommandId.NAPARI_INFO.command_title,
        callback=_show_about,
        menus=[{'id': MenuId.MENUBAR_HELP, 'group': MenuGroup.RENDER}],
        status_tip=trans._('About napari'),
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.Slash)],
    ),
    Action(
        id=CommandId.NAPARI_ABOUT_MACOS,
        title=CommandId.NAPARI_ABOUT_MACOS.command_title,
        callback=_show_about,
        menus=[
            {
                'id': MenuId.MENUBAR_HELP,
                'group': MenuGroup.RENDER,
                'when': sys.platform == 'darwin',
            }
        ],
        status_tip=trans._('About napari'),
    ),
]

if ask_opt_in is not None:
    Q_HELP_ACTIONS.append(
        Action(
            id=CommandId.TOGGLE_BUG_REPORT_OPT_IN,
            title=CommandId.TOGGLE_BUG_REPORT_OPT_IN.command_title,
            callback=lambda: ask_opt_in(force=True),
            menus=[{'id': MenuId.MENUBAR_HELP}],
        )
    )
