
"""
When calling vspscript.exe to execute wing_openvsp.vspscript and generate  wing_openvsp_DegenGeom.csv an error is
sometimes risen, stoping the optim without graceful way to recover. As re-directed from the batch file it is related to
OpenVSP:

VSPAERO viewer not found.
VSPAERO slicer not found.
Internal error in segmentintersection():
  Topological inconsistency after splitting a segment.
  Please report this bug to jrs@cs.berkeley.edu
  Include the message above, your input data set, and the exact
    command line you used to run Triangle.

As per Florent, this may be related to the modification of wing_openvsp.vspscript to increase the wing sections.
As suggested in https://groups.google.com/g/openvsp/c/-fzvGc9NTIU  modifying slightly values in wing_openvsp.vspscript,
could solve the problem. For instance, in some tests a file triggering the error worked fine after changing the 15th
decimal of "x_wing".
The simplest work around consists in:
1) Rounding values written in wing_openvsp.vspscript
2) Changing non-significative decimals
"""


class WingFileModifying:

    def __init__(self):
        """
        Initialize attributes.
        """
        self._filename = None

    def set_file(self, filename):
        """
        Set the name of the file to be changed.

        The original data of the file is also read into memory when this method is called.

        Parameters
        ----------
        filename : str
            Name of the template file to be used.
        """
        self._filename = filename
        self._count = 1

        templatefile = open(filename, 'r')
        self._data = templatefile.readlines()
        templatefile.close()

    def replace_values(self, line):
        """
        Replace the values in a certain line.

        Parameters
        ----------
        line : int
            Number of the line to change
        """

        # Takes the value of the demanded line
        string = self._data[line].split()[4]

        # Count the number of decimals
        number_decimals = len(string) - (string.find(".") + 1)

        # Takes the number and modifies it by adding 10 ** number of decimals
        new_value = round(float(float(string) + self._count * (10 ** -(int(number_decimals)))), number_decimals)

        # Copies self._data in filedata_lines and eplace the new value
        filedata_lines = self._data
        filedata_lines[line] = filedata_lines[line].replace(string, str(new_value))

        # Once replaced, we open the file and write this
        with open(self._filename, "w") as file:
            file.writelines(filedata_lines)
        file.close()

        # Increases the count on self._count for next call
        self._count += 1

    def reinitialize_file(self):
        """
        Rewrites the input file with the original content.
        Sets the count again to 1
        """

        with open(self._filename, "w") as file:
            file.writelines(self._data)
        file.close()
        self._count = 1












