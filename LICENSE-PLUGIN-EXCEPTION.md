# Additional permission for plugins

n0face is licensed under the GNU Affero General Public License, version 3 or later. As an
additional permission under section 7 of that license, the copyright holder grants the following:

You may combine n0face with a plugin, and convey the combination, without the plugin being
subject to the AGPL, provided that the plugin interacts with n0face only through the plugin
interface defined in `src/n0face/plugins.py` (registering masks, overlays, pages and features in
`n0face.plugins.registry`, and using the public modules those registrations need). n0face itself,
including any modifications you make to it, remains under the AGPL.

"Plugin" means a separately distributed Python package that n0face loads through that
interface (for example, n0face PRO). This permission does not apply to code copied from n0face
into the plugin.

You may remove this additional permission from copies or modified versions of n0face that you
convey, as described in section 7.
