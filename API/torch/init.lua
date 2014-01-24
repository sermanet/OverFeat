require 'liboverfeat_torch'

-- Note: overfeat works with FloatTensor only

overfeat = {}

function overfeat.init(path_to_weight, net_idx)
   net_idx = net_idx or 0
   overfeat.input = torch.FloatTensor()
   liboverfeat_torch.init(path_to_weight, net_idx)
end

function overfeat.free()
   liboverfeat_torch.free()
end

-- The user can provide a output tensor so it is not reallocated
function overfeat.fprop(input, output)
   overfeat.input:resizeAs(input)
   overfeat.input:copy(input)
   overfeat.input = overfeat.input:mul(255)
   output = output or torch.FloatTensor()
   liboverfeat_torch.fprop(overfeat.input, output)
   return output
end

-- The user can provide a output tensor so it is not reallocated
function overfeat.get_output(i, output)
   output = output or torch.FloatTensor()
   liboverfeat_torch.get_output(output, i)
   return output
end

function overfeat.get_n_layers()
   return liboverfeat_torch.get_n_layers()
end

function overfeat.get_class_name(i)
   return liboverfeat_torch.get_class_name(i)
end