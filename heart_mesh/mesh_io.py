import vtk

def replace_class_label(input_surface, output_surface, from_label=5, to_label=2):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_surface)
    reader.Update()
    polydata = reader.GetOutput()

    class_array = polydata.GetPointData().GetArray("class")
    if class_array is None:
        raise ValueError("The 'class' array was not found in the point data of the input file.")

    for i in range(polydata.GetNumberOfPoints()):
        if class_array.GetValue(i) == from_label:
            class_array.SetValue(i, to_label)
    class_array.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_surface)
    writer.SetInputData(polydata)
    writer.Write()
