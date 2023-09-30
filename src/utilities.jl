function Base.show(io::IO, ::MIME"text/plain", model::SSM1D)
    return _show(io::IO, model)
end

function Base.show(io::IO, ::MIME"text/plain", model::SSM2D)
    return _show(io::IO, model)
end

function Base.show(io::IO, ::MIME"text/plain", model::ContinuousMultivariateSSM)
    return _show(io::IO, model)
end

function _show(io::IO, model)
    values = [getfield(model, f) for f in fieldnames(typeof(model))]
    values = map(x -> typeof(x) == Bool ? string(x) : x, values)
    T = typeof(model)
    model_name = string(T.name.name)
    return pretty_table(io,
        values;
        title=model_name,
        row_name_column_title="Parameter",
        compact_printing=false,
        header=["Value"],
        row_name_alignment=:l,
        row_names=[fieldnames(typeof(model))...],
        formatters=ft_printf("%5.2f"),
        alignment=:l,
    )
end